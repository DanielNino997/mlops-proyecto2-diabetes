"""
Pruebas de src/train.py.

Usan MLflow con una base SQLite temporal, solo para pruebas (la entrega real
usa PostgreSQL y MinIO), y datos sintéticos de tests/fake_data.py.

    python3 -m pytest -q tests/test_train.py
"""

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

import config
import fake_data
import preprocessing
import schemas
import train


# ---------------------------------------------------------------------------
# Métricas y umbral (no necesitan MLflow)
# ---------------------------------------------------------------------------
def test_compute_metrics_on_a_known_case():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.3, 0.7, 0.2, 0.1, 0.1, 0.05]
    m = train.compute_metrics(y, p, 0.5)       # predice 1 en 0.9, 0.8, 0.7
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 4)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["f2"] == pytest.approx(2 / 3)     # con P = R todos los F-beta coinciden
    assert m["specificity"] == pytest.approx(0.8)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert 0 < m["pr_auc"] <= 1 and 0 < m["roc_auc"] <= 1


def test_f2_weighs_recall_more_than_precision():
    # precisión 0.5 / recall 1.0  vs  precisión 1.0 / recall 0.5
    y = [1, 1, 0, 0]
    high_recall = train.compute_metrics(y, [0.9, 0.8, 0.7, 0.1], 0.5)    # P=.67 R=1
    high_precision = train.compute_metrics(y, [0.9, 0.2, 0.1, 0.1], 0.5)  # P=1 R=.5
    assert high_recall["f2"] > high_precision["f2"]


def test_best_threshold_prefers_recall_and_is_reproducible():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.11).astype(int)
    p = np.clip(rng.normal(0.25 + 0.2 * y, 0.15), 0, 1)
    t2 = train.best_threshold(y, p, beta=2.0)
    t1 = train.best_threshold(y, p, beta=1.0)
    assert t2 <= t1  # beta mayor -> umbral más bajo (más recall)
    assert train.best_threshold(y, p, beta=2.0) == t2
    # El umbral elegido da el mejor F2 posible entre una malla de umbrales.
    best = train.compute_metrics(y, p, t2)["f2"]
    grid = max(train.compute_metrics(y, p, t)["f2"] for t in np.linspace(0.01, 0.99, 99))
    assert best >= grid - 1e-9


def test_primary_value_rejects_unknown_metric():
    with pytest.raises(ValueError, match="no existe"):
        train.primary_value({"f2": 0.5}, "accuracy")


def test_features_frame_dtypes_and_order():
    raw = fake_data.make_fake_dataframe(300, seed=1)
    clean = preprocessing.clean_data(raw)
    frame = train.features_frame(clean)
    assert list(frame.columns) == schemas.FEATURE_COLUMNS
    for column in schemas.NUMERIC_FEATURES:
        assert frame[column].dtype == "int64"
    for column in schemas.CATEGORICAL_FEATURES:
        assert frame[column].map(type).eq(str).all()


def test_pipeline_has_one_hot_inside_and_handles_unseen_categories():
    clean = preprocessing.clean_data(fake_data.make_fake_dataframe(1500, seed=2))
    X = train.features_frame(clean)
    y = clean[config.TARGET_COLUMN]
    pipe = train.build_pipeline(LogisticRegression(max_iter=300)).fit(X, y)

    odd = X.head(3).copy()
    odd["admission_source_id"] = "999"      # código nunca visto en el entrenamiento
    odd["race"] = "Martian"
    proba = pipe.predict_proba(odd)[:, 1]
    assert proba.shape == (3,) and np.isfinite(proba).all()


def test_check_splits_errors():
    clean = preprocessing.clean_data(fake_data.make_fake_dataframe(3000, seed=3))
    splits = preprocessing.split_dataset(clean)
    train.check_splits(splits)  # el caso normal no falla

    with pytest.raises(ValueError, match="vacío"):
        train.check_splits({**splits, "val": splits["val"].iloc[0:0]})
    few = splits["test"].copy()
    few[config.TARGET_COLUMN] = 0
    with pytest.raises(ValueError, match="positivos"):
        train.check_splits({**splits, "test": few})


# ---------------------------------------------------------------------------
# Con MLflow
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def splits():
    clean = preprocessing.clean_data(fake_data.make_fake_dataframe(6000, seed=11))
    return preprocessing.split_dataset(clean)


@pytest.fixture()
def mlflow_store(tmp_path):
    train.configure_mlflow(f"sqlite:///{tmp_path / 'mlflow.db'}", "pruebas")
    yield MlflowClient()


def test_train_candidate_logs_everything(splits, mlflow_store):
    result = train.train_candidate("logistic_regression", splits, tags={"batch_id": "1"})
    run = mlflow_store.get_run(result["run_id"])

    assert run.data.tags["candidate"] == "logistic_regression"
    assert run.data.tags["batch_id"] == "1"
    for split in ("train", "val", "test"):
        for metric in ("f2", "recall", "precision", "pr_auc", "roc_auc", "threshold"):
            assert f"{split}_{metric}" in run.data.metrics, f"{split}_{metric}"
    assert run.data.metrics["primary_metric_val"] == pytest.approx(result["val_primary"])
    assert "model__C" in run.data.params and run.data.params["n_features"] == str(len(schemas.FEATURE_COLUMNS))

    artifacts = {a.path for a in mlflow_store.list_artifacts(result["run_id"], "evaluation")}
    assert {"evaluation/confusion_matrix_test.png", "evaluation/pr_curve.png",
            "evaluation/training_summary.json", "evaluation/features.json"} <= artifacts
    assert result["model_uri"]

    # El umbral se eligió en validación, no en prueba.
    assert result["metrics"]["val"]["threshold"] == result["metrics"]["test"]["threshold"]
    # La clase positiva se detecta mejor que al azar.
    assert result["metrics"]["test"]["roc_auc"] > 0.6


def test_select_best_uses_validation_not_test():
    a = {"name": "a", "val_primary": 0.50, "test_primary": 0.90}
    b = {"name": "b", "val_primary": 0.60, "test_primary": 0.10}
    assert train.select_best([a, b])["name"] == "b"
    with pytest.raises(ValueError):
        train.select_best([])


def test_first_model_becomes_champion_and_api_can_load_it(splits, mlflow_store):
    outcome = train.train_all(splits, ["logistic_regression", "hist_gradient_boosting"])
    decision = outcome["decision"]
    assert decision["promoted"] is True and decision["champion_version"] is None

    champion = mlflow_store.get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS)
    assert int(champion.version) == decision["version"] == 1
    assert float(champion.tags["threshold"]) == pytest.approx(
        [r for r in outcome["results"] if r["name"] == outcome["best"]][0]["threshold"], abs=1e-6)
    assert champion.tags["promoted"] == "true"

    # Lo que hace la API: cargar por alias y predecir sobre el formato del contrato.
    import mlflow.sklearn
    model = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}")
    example = pd.DataFrame([schemas.EXAMPLE_PATIENT.model_dump()])
    proba = model.predict_proba(example)[:, 1]
    assert 0 <= float(proba[0]) <= 1


def test_tie_keeps_the_champion(splits, mlflow_store):
    train.train_all(splits, ["logistic_regression"])
    second = train.train_all(splits, ["logistic_regression"])["decision"]
    assert second["promoted"] is False                   # mismo modelo, mismo dato: empate
    assert second["champion_test"] == pytest.approx(second["candidate_test"])
    alias = mlflow_store.get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS)
    assert int(alias.version) == 1
    v2 = mlflow_store.get_model_version(config.MODEL_NAME, "2")
    assert v2.tags["promoted"] == "false"
    run = mlflow_store.get_run(v2.run_id)
    assert run.data.tags["promoted"] == "false" and "se conserva el champion" in run.data.tags["promotion_reason"]


def test_better_candidate_replaces_champion_and_worse_does_not(splits, mlflow_store, monkeypatch):
    weak = {"weak": DummyClassifier(strategy="prior")}
    monkeypatch.setattr(train, "candidate_estimators", lambda: weak)
    first = train.train_all(splits, ["weak"])["decision"]
    assert first["promoted"] is True and first["version"] == 1

    monkeypatch.undo()                                    # vuelven los modelos reales
    better = train.train_all(splits, ["logistic_regression"])["decision"]
    assert better["promoted"] is True and better["version"] == 2
    assert better["candidate_test"] > better["champion_test"]
    assert int(mlflow_store.get_model_version_by_alias(
        config.MODEL_NAME, config.MODEL_ALIAS).version) == 2

    monkeypatch.setattr(train, "candidate_estimators", lambda: weak)
    worse = train.train_all(splits, ["weak"])["decision"]
    assert worse["promoted"] is False
    assert int(mlflow_store.get_model_version_by_alias(
        config.MODEL_NAME, config.MODEL_ALIAS).version) == 2


def test_min_improvement_blocks_marginal_gains(splits, mlflow_store, monkeypatch):
    train.train_all(splits, ["logistic_regression"])
    monkeypatch.setattr(config, "MIN_IMPROVEMENT", 0.5)   # exige una mejora enorme
    monkeypatch.setattr(train, "candidate_estimators", lambda: {
        "logistic_regression": LogisticRegression(C=0.5, max_iter=500, class_weight="balanced")})
    decision = train.train_all(splits, ["logistic_regression"])["decision"]
    assert decision["promoted"] is False


def test_comparison_refuses_a_different_test_set(splits, mlflow_store):
    result = train.train_candidate("logistic_regression", splits)
    other_test = splits["test"].iloc[:-5]                 # llegaron/salieron datos
    with pytest.raises(ValueError, match="no sería justa"):
        train.register_and_promote(result, other_test)


def test_unreadable_champion_does_not_block_promotion(splits, mlflow_store, monkeypatch):
    train.train_all(splits, ["logistic_regression"])

    def broken(*args, **kwargs):
        raise RuntimeError("artefacto dañado")

    monkeypatch.setattr(train, "evaluate_champion", broken)
    decision = train.train_all(splits, ["logistic_regression"])["decision"]
    assert decision["promoted"] is True and decision["champion_test"] is None
    assert "no se pudo evaluar" in decision["reason"]


def test_cli_with_csv(tmp_path, capsys):
    csv = fake_data.write_fake_csv(tmp_path / "datos.csv", n=4000, seed=21)
    code = train.main(["--csv", str(csv), "--tracking-uri", f"sqlite:///{tmp_path / 'm.db'}",
                       "--models", "logistic_regression,hist_gradient_boosting"])
    out = capsys.readouterr().out
    assert code == 0
    assert "logistic_regression" in out and "promovido" in out
