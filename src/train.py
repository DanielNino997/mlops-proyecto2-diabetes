"""
Entrenamiento, comparación y promoción del modelo (carril B).

Qué hace, en orden:

  1. Entrena varios modelos candidatos. Cada uno es un Pipeline de sklearn
     que incluye el one-hot y el escalado, así la API recibe valores "humanos"
     y nunca repite transformaciones.
  2. Para cada candidato elige el umbral de decisión en VALIDACIÓN (el que
     maximiza F-beta, con beta = 2: el recall pesa el doble que la precisión),
     y mide train, validación y prueba.
  3. Registra cada candidato como una corrida de MLflow: parámetros, métricas,
     gráficas, resumen y el modelo serializado.
  4. Elige el mejor candidato por la métrica principal en VALIDACIÓN.
  5. Lo registra como nueva versión del modelo y lo compara contra el
     champion actual en el MISMO conjunto de prueba. Solo si lo supera,
     mueve el alias "champion" a la nueva versión.

Por qué se compara en el mismo conjunto de prueba: cada ejecución del DAG
agrega lotes y el conjunto de prueba crece. Comparar la métrica nueva con la
que se guardó cuando se entrenó el champion sería injusto; por eso el champion
se vuelve a evaluar hoy, sobre los mismos datos que el candidato.

Uso desde la terminal (sin Airflow):

    # con los datos de PostgreSQL (clean_data ya poblada por el DAG)
    python3 src/train.py

    # con el CSV directamente, para probar en local
    python3 src/train.py --csv data/Diabetes/Diabetes.csv --tracking-uri http://localhost:5000

Las funciones train_candidate() y register_and_promote() son las que llama el
DAG, una tarea por modelo y una tarea final de selección.
"""

import argparse
import hashlib
import inspect
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # sin pantalla: el entrenamiento corre dentro de un pod
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
import preprocessing
import schemas

logger = logging.getLogger(__name__)

MODEL_ARTIFACT_PATH = "model"
SPLITS = ("train", "val", "test")
METRIC_NAMES = ("f2", "recall", "f1", "pr_auc", "roc_auc")


# ---------------------------------------------------------------------------
# Datos -> matriz de entrada del modelo
# ---------------------------------------------------------------------------
def features_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona las 24 features del contrato, en orden y con tipos fijos.

    Los números van como int64 y las categorías como texto, igual que cuando la
    API arma el DataFrame desde el JSON; así entrenamiento e inferencia ven
    exactamente lo mismo.
    """
    frame = df[schemas.FEATURE_COLUMNS].copy()
    for column in schemas.NUMERIC_FEATURES:
        frame[column] = frame[column].astype("int64")
    for column in schemas.CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype(str)
    return frame


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Huella de los encuentros de un conjunto; sirve para comprobar que dos
    evaluaciones usaron exactamente los mismos datos."""
    ids = ",".join(str(i) for i in sorted(df["encounter_id"].astype("int64")))
    return hashlib.sha256(ids.encode()).hexdigest()[:16]


def check_splits(splits: Dict[str, pd.DataFrame]) -> None:
    """Falla con un mensaje claro si falta algún conjunto o hay muy pocos positivos."""
    for name in SPLITS:
        if name not in splits or len(splits[name]) == 0:
            raise ValueError(f"El conjunto '{name}' está vacío; no se puede entrenar")
        positives = int(splits[name][config.TARGET_COLUMN].sum())
        if positives < config.MIN_POSITIVES_PER_SPLIT:
            raise ValueError(
                f"El conjunto '{name}' tiene solo {positives} positivos "
                f"(mínimo {config.MIN_POSITIVES_PER_SPLIT}); faltan más lotes de datos"
            )


# ---------------------------------------------------------------------------
# Modelos candidatos
# ---------------------------------------------------------------------------
def candidate_estimators() -> Dict[str, Any]:
    """
    Tres familias de modelos. Todas compensan el desbalance (~11% de
    positivos) con class_weight; el umbral se ajusta después en validación.
    """
    seed = config.RANDOM_STATE
    return {
        "logistic_regression": LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=20,
            class_weight="balanced_subsample", n_jobs=config.TRAIN_N_JOBS,
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08, max_depth=6, max_iter=300,
            class_weight="balanced", early_stopping=True, n_iter_no_change=15,
            random_state=seed,
        ),
    }


def build_pipeline(estimator: Any) -> Pipeline:
    """Preprocesamiento (escalado + one-hot) y modelo en un solo objeto."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), schemas.MODEL_NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             schemas.MODEL_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


# ---------------------------------------------------------------------------
# Métricas y umbral
# ---------------------------------------------------------------------------
def best_threshold(y_true, proba, beta: float = config.THRESHOLD_BETA) -> float:
    """Umbral que maximiza F-beta sobre los datos dados (se usa en validación)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]  # alinear con thresholds
    b2 = beta ** 2
    denominator = b2 * precision + recall
    fbeta = np.divide((1 + b2) * precision * recall, denominator,
                      out=np.zeros_like(denominator), where=denominator > 0)
    return float(thresholds[int(np.argmax(fbeta))])


def compute_metrics(y_true, proba, threshold: float) -> Dict[str, float]:
    """Métricas de clasificación a un umbral dado, más PR-AUC y ROC-AUC (sin umbral)."""
    y_true = np.asarray(y_true)
    predicted = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    def safe(numerator, denominator):
        return float(numerator / denominator) if denominator else 0.0

    precision = safe(tp, tp + fp)
    recall = safe(tp, tp + fn)

    def f_beta(beta):
        b2 = beta ** 2
        return safe((1 + b2) * precision * recall, b2 * precision + recall)

    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f_beta(1.0),
        "f2": f_beta(2.0),
        "specificity": safe(tn, tn + fp),
        "accuracy": safe(tp + tn, len(y_true)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "predicted_positive_rate": float(predicted.mean()),
        "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
    }


def primary_value(metrics: Dict[str, float], metric: str = config.PRIMARY_METRIC) -> float:
    if metric not in METRIC_NAMES:
        raise ValueError(f"PRIMARY_METRIC '{metric}' no existe; usa una de {METRIC_NAMES}")
    return float(metrics[metric])


# ---------------------------------------------------------------------------
# Gráficas y artefactos
# ---------------------------------------------------------------------------
def _plot_confusion(metrics: Dict[str, float], path: Path, title: str) -> None:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["No <30", "Sí <30"])
    ax.set_yticks([0, 1], ["No <30", "Sí <30"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(title)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{int(value):,}", ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_pr_curves(curves: Dict[str, Any], thresholds: Dict[str, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for name, (y_true, proba) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ax.plot(recall, precision, label=f"{name} (PR-AUC {average_precision_score(y_true, proba):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisión")
    ax.set_title("Curva precisión-recall")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _scalar_params(estimator: Any) -> Dict[str, Any]:
    """Hiperparámetros simples (números, texto, booleanos) del modelo."""
    return {
        f"model__{key}": value
        for key, value in estimator.get_params().items()
        if value is None or isinstance(value, (int, float, str, bool))
    }


# ---------------------------------------------------------------------------
# Entrenamiento de un candidato
# ---------------------------------------------------------------------------
def configure_mlflow(tracking_uri: Optional[str] = None, experiment: Optional[str] = None) -> None:
    mlflow.set_tracking_uri(tracking_uri or config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment or config.EXPERIMENT_NAME)


def train_candidate(
    name: str,
    splits: Dict[str, pd.DataFrame],
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Entrena un candidato y lo registra como una corrida de MLflow.

    Devuelve un diccionario simple (serializable a JSON) con el id de la
    corrida, el umbral y las métricas; es lo que el DAG pasa entre tareas.
    Requiere que configure_mlflow() ya se haya llamado.
    """
    estimators = candidate_estimators()
    if name not in estimators:
        raise ValueError(f"Modelo desconocido '{name}'; opciones: {sorted(estimators)}")
    check_splits(splits)

    X = {s: features_frame(splits[s]) for s in SPLITS}
    y = {s: splits[s][config.TARGET_COLUMN].astype(int).to_numpy() for s in SPLITS}

    pipeline = build_pipeline(estimators[name])
    started = time.time()
    pipeline.fit(X["train"], y["train"])
    train_seconds = time.time() - started

    proba = {s: pipeline.predict_proba(X[s])[:, 1] for s in SPLITS}
    threshold = best_threshold(y["val"], proba["val"])
    metrics = {s: compute_metrics(y[s], proba[s], threshold) for s in SPLITS}

    result: Dict[str, Any] = {
        "name": name,
        "threshold": threshold,
        "metrics": metrics,
        "primary_metric": config.PRIMARY_METRIC,
        "val_primary": primary_value(metrics["val"]),
        "test_primary": primary_value(metrics["test"]),
        "test_fingerprint": dataset_fingerprint(splits["test"]),
        "rows": {s: int(len(splits[s])) for s in SPLITS},
        "positives": {s: int(y[s].sum()) for s in SPLITS},
        "train_seconds": round(train_seconds, 2),
    }

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({
            "candidate": name,
            "primary_metric": config.PRIMARY_METRIC,
            "test_fingerprint": result["test_fingerprint"],
            **(tags or {}),
        })
        mlflow.log_params({
            "model_name": name,
            "n_features": len(schemas.FEATURE_COLUMNS),
            "threshold_beta": config.THRESHOLD_BETA,
            "random_state": config.RANDOM_STATE,
            "rows_train": result["rows"]["train"],
            "rows_val": result["rows"]["val"],
            "rows_test": result["rows"]["test"],
            **_scalar_params(pipeline.named_steps["model"]),
        })
        for split in SPLITS:
            mlflow.log_metrics({f"{split}_{k}": v for k, v in metrics[split].items()})
        mlflow.log_metric("train_seconds", train_seconds)
        mlflow.log_metric("primary_metric_val", result["val_primary"])
        mlflow.log_metric("primary_metric_test", result["test_primary"])

        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            _plot_confusion(metrics["test"], folder_path / "confusion_matrix_test.png",
                            f"{name}: prueba (umbral {threshold:.3f})")
            _plot_pr_curves({"validación": (y["val"], proba["val"]),
                             "prueba": (y["test"], proba["test"])},
                            {"umbral": threshold}, folder_path / "pr_curve.png")
            (folder_path / "training_summary.json").write_text(
                json.dumps({k: v for k, v in result.items()}, indent=2, ensure_ascii=False),
                encoding="utf-8")
            (folder_path / "features.json").write_text(
                json.dumps({"numeric": schemas.MODEL_NUMERIC_FEATURES,
                            "categorical": schemas.MODEL_CATEGORICAL_FEATURES},
                           indent=2), encoding="utf-8")
            mlflow.log_artifacts(str(folder_path), artifact_path="evaluation")

        sample = X["train"].head(5)
        # MLflow 3 renombró artifact_path a name; se usa el que exista para no
        # depender de la versión del cliente.
        name_key = "name" if "name" in inspect.signature(mlflow.sklearn.log_model).parameters \
            else "artifact_path"
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            signature=infer_signature(sample, pipeline.predict_proba(sample)[:, 1]),
            input_example=sample,
            **{name_key: MODEL_ARTIFACT_PATH},
        )
        result["run_id"] = run.info.run_id
        result["model_uri"] = model_info.model_uri

    logger.info("Candidato %s: val_%s=%.4f test_%s=%.4f umbral=%.3f (%.1fs)",
                name, config.PRIMARY_METRIC, result["val_primary"], config.PRIMARY_METRIC,
                result["test_primary"], threshold, train_seconds)
    return result


def select_best(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """El mejor candidato según la métrica principal en VALIDACIÓN (no en prueba)."""
    if not results:
        raise ValueError("No hay candidatos para comparar")
    return max(results, key=lambda r: (r["val_primary"], r["name"]))


# ---------------------------------------------------------------------------
# Registro y promoción
# ---------------------------------------------------------------------------
def _is_missing(error: MlflowException) -> bool:
    """True si el error significa "ese modelo o alias no existe todavía"."""
    text = str(error).lower()
    return (
        getattr(error, "error_code", "") == "RESOURCE_DOES_NOT_EXIST"
        or "does not exist" in text
        or "not found" in text
    )


def get_champion(client: MlflowClient, model_name: str, alias: str):
    """Versión que hoy tiene el alias, o None si todavía no hay ninguna."""
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except MlflowException as error:
        if _is_missing(error):
            return None
        raise


def evaluate_champion(champion, test_df: pd.DataFrame) -> Dict[str, float]:
    """Evalúa el champion actual sobre el conjunto de prueba de hoy."""
    model = mlflow.sklearn.load_model(f"models:/{champion.name}/{champion.version}")
    threshold = float((champion.tags or {}).get("threshold", config.DECISION_THRESHOLD))
    proba = model.predict_proba(features_frame(test_df))[:, 1]
    return compute_metrics(test_df[config.TARGET_COLUMN].astype(int), proba, threshold)


def register_and_promote(
    best: Dict[str, Any],
    test_df: pd.DataFrame,
    model_name: str = config.MODEL_NAME,
    alias: str = config.MODEL_ALIAS,
) -> Dict[str, Any]:
    """
    Registra el mejor candidato y lo promueve a champion solo si supera al
    champion actual en el mismo conjunto de prueba.
    """
    metric = best["primary_metric"]
    if dataset_fingerprint(test_df) != best["test_fingerprint"]:
        raise ValueError(
            "El conjunto de prueba cambió entre el entrenamiento y la comparación; "
            "la comparación no sería justa. Vuelve a ejecutar el entrenamiento."
        )

    client = MlflowClient()
    champion = get_champion(client, model_name, alias)

    champion_value: Optional[float] = None
    champion_note = "no había champion"
    if champion is not None:
        try:
            champion_metrics = evaluate_champion(champion, test_df)
            champion_value = primary_value(champion_metrics, metric)
            champion_note = f"champion v{champion.version}"
        except Exception as error:  # un champion ilegible no debe bloquear el sistema
            logger.warning("No se pudo evaluar el champion v%s: %s", champion.version, error)
            champion_note = f"champion v{champion.version} no se pudo evaluar ({error})"

    candidate_value = float(best["test_primary"])
    promote = champion_value is None or candidate_value > champion_value + config.MIN_IMPROVEMENT

    version = mlflow.register_model(best["model_uri"], model_name)
    version_tags = {
        "threshold": f"{best['threshold']:.6f}",
        "candidate": best["name"],
        "primary_metric": metric,
        f"test_{metric}": f"{candidate_value:.4f}",
        f"val_{metric}": f"{best['val_primary']:.4f}",
        "rows_train": str(best["rows"]["train"]),
        "rows_test": str(best["rows"]["test"]),
        "test_fingerprint": best["test_fingerprint"],
        "promoted": str(bool(promote)).lower(),
    }
    for key, value in version_tags.items():
        client.set_model_version_tag(model_name, version.version, key, value)

    if promote:
        client.set_registered_model_alias(model_name, alias, version.version)

    reason = (
        f"{best['name']} v{version.version}: {metric}={candidate_value:.4f} en prueba "
        + (f"vs {champion_value:.4f} del {champion_note}" if champion_value is not None
           else f"({champion_note})")
        + (f" -> promovido a '{alias}'" if promote else f" -> se conserva el champion v{champion.version}")
    )
    client.set_tag(best["run_id"], "registered_version", str(version.version))
    client.set_tag(best["run_id"], "promoted", str(bool(promote)).lower())
    client.set_tag(best["run_id"], "promotion_reason", reason)
    if champion_value is not None:
        client.log_metric(best["run_id"], f"champion_test_{metric}", champion_value)

    decision = {
        "model_name": model_name,
        "alias": alias,
        "version": int(version.version),
        "candidate": best["name"],
        "metric": metric,
        "candidate_test": candidate_value,
        "champion_version": int(champion.version) if champion is not None else None,
        "champion_test": champion_value,
        "promoted": bool(promote),
        "reason": reason,
    }
    logger.info("Decisión de promoción: %s", reason)
    return decision


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------
def train_all(
    splits: Dict[str, pd.DataFrame],
    models: Optional[List[str]] = None,
    tags: Optional[Dict[str, str]] = None,
    promote: bool = True,
) -> Dict[str, Any]:
    """Entrena todos los candidatos, elige el mejor y lo promueve si corresponde."""
    names = models or list(candidate_estimators())
    results = [train_candidate(name, splits, tags) for name in names]
    best = select_best(results)
    decision = register_and_promote(best, splits["test"]) if promote else None
    return {"results": results, "best": best["name"], "decision": decision}


def _print_report(outcome: Dict[str, Any]) -> None:
    print(f"\n{'modelo':26s}{'val_' + config.PRIMARY_METRIC:>10s}{'test_' + config.PRIMARY_METRIC:>10s}"
          f"{'recall':>9s}{'precisión':>11s}{'PR-AUC':>9s}{'umbral':>9s}")
    for r in outcome["results"]:
        t = r["metrics"]["test"]
        mark = "  <- mejor" if r["name"] == outcome["best"] else ""
        print(f"{r['name']:26s}{r['val_primary']:>10.4f}{r['test_primary']:>10.4f}"
              f"{t['recall']:>9.3f}{t['precision']:>11.3f}{t['pr_auc']:>9.3f}{t['threshold']:>9.3f}{mark}")
    if outcome["decision"]:
        print("\n" + outcome["decision"]["reason"])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Entrena, registra y promueve el modelo")
    parser.add_argument("--csv", help="CSV fuente; si se omite se lee clean_data de PostgreSQL")
    parser.add_argument("--tracking-uri", help="URI de MLflow (por defecto MLFLOW_TRACKING_URI)")
    parser.add_argument("--experiment", help="Nombre del experimento")
    parser.add_argument("--models", help="Lista separada por comas; por defecto todos")
    parser.add_argument("--no-promote", action="store_true", help="Solo entrenar y registrar corridas")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_mlflow(args.tracking_uri, args.experiment)

    if args.csv:
        raw = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
        clean = preprocessing.clean_data(raw)
        splits = preprocessing.split_dataset(clean)
    else:
        import db  # solo aquí: el DAG y la terminal con CSV no necesitan psycopg2

        splits = db.read_splits(db.get_engine())

    models = [m.strip() for m in args.models.split(",")] if args.models else None
    outcome = train_all(splits, models, promote=not args.no_promote)
    _print_report(outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
