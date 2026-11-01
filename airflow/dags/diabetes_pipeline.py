"""
DAG del proyecto: lotes -> raw_data -> limpieza -> clean_data -> entrenamiento -> champion.

Cada ejecución procesa UN lote de BATCH_SIZE filas del CSV (config.BATCH_SIZE):

  init_schema -> check_source -> plan_batch -> load_raw -> validate_raw
              -> preprocess -> assign_splits -> train_<modelo> (en paralelo)
              -> select_and_promote

Cuando el archivo ya no tiene filas nuevas, plan_batch avisa y las tareas
siguientes se marcan "skipped": ejecutar el DAG de más no rompe nada.
Todas las tareas son idempotentes (ver src/db.py), así que un reintento no
duplica datos.

Variables de entorno: las mismas de src/config.py (POSTGRES_*, MLFLOW_*, ...)
más DAG_SCHEDULE (por defecto "@daily").
"""

import logging
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG

# Los imports de los operadores cambian entre Airflow 2 y 3.
try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:  # Airflow 2.x
    from airflow.operators.python import PythonOperator

try:
    from airflow.exceptions import AirflowSkipException
except ImportError:  # Airflow 3 (SDK)
    from airflow.sdk.exceptions import AirflowSkipException  # type: ignore

# src/ (config, db, preprocessing, train) se copia a /opt/airflow/src en la imagen.
for _candidate in (os.getenv("PROJECT_SRC", ""), "/opt/airflow/src",
                   os.path.join(os.path.dirname(__file__), "..", "..", "src")):
    if _candidate and os.path.isdir(_candidate):
        sys.path.insert(0, os.path.abspath(_candidate))
        break

import config  # noqa: E402

log = logging.getLogger(__name__)

MODELS = ["logistic_regression", "random_forest", "hist_gradient_boosting"]


# ---------------------------------------------------------------------------
# Tareas. Los imports pesados van dentro de cada función para que Airflow
# pueda leer el DAG rápido (parseo) sin cargar pandas ni sklearn.
# ---------------------------------------------------------------------------
def _batch(ti):
    """El lote planeado por plan_batch; si no hay, no hay nada que hacer."""
    batch = ti.xcom_pull(task_ids="plan_batch")
    if not batch:
        raise AirflowSkipException("No hay filas nuevas en el archivo fuente.")
    return batch


def _fail_batch(engine, batch_id, error):
    import db
    try:
        db.set_batch_status(engine, batch_id, "failed", error=str(error)[:500])
    except Exception:  # no ocultar el error original
        log.exception("No se pudo marcar el lote %s como failed", batch_id)


def init_schema():
    import db
    db.init_schema(db.get_engine())


def check_source():
    import db
    info = db.check_source_file(config.SOURCE_FILE)
    log.info("Archivo fuente: %s", info)
    return {"rows": int(info["rows"])}


def plan_batch(ti, run_id):
    import db
    total = ti.xcom_pull(task_ids="check_source")["rows"]
    batch = db.plan_batch(db.get_engine(), run_id, config.SOURCE_FILE, total, config.BATCH_SIZE)
    if batch is None:
        log.info("El archivo ya se cargó completo (%s filas).", total)
        return None
    # Solo valores simples: XCom no guarda fechas.
    keep = ("batch_id", "row_start", "row_end", "resumed", "total_rows")
    summary = {k: batch[k] for k in keep}
    log.info("Lote a procesar: %s", summary)
    return summary


def load_raw(ti):
    import db
    batch = _batch(ti)
    engine = db.get_engine()
    try:
        source = db.read_source_file(config.SOURCE_FILE)
        return db.load_raw_batch(engine, {**batch, "source_file": config.SOURCE_FILE}, source)
    except Exception as error:
        _fail_batch(engine, batch["batch_id"], error)
        raise


def validate_raw(ti):
    import db
    batch = _batch(ti)
    engine = db.get_engine()
    try:
        report = db.validate_batch(engine, batch["batch_id"])
    except Exception as error:
        _fail_batch(engine, batch["batch_id"], error)
        raise
    log.info("Validación del lote: %s", report)
    return {k: v for k, v in report.items() if isinstance(v, (int, float, str, bool))}


def preprocess(ti):
    import db
    batch = _batch(ti)
    engine = db.get_engine()
    try:
        report = db.process_batch(engine, batch["batch_id"])
    except Exception as error:
        _fail_batch(engine, batch["batch_id"], error)
        raise
    log.info("Limpieza del lote: %s", report)
    return {k: v for k, v in report.items() if isinstance(v, (int, float, str, bool))}


def assign_splits(ti):
    import db
    _batch(ti)
    engine = db.get_engine()
    db.assign_splits(engine)
    summary = db.split_summary(engine)
    log.info("Particiones: %s", summary)
    return {k: v for k, v in summary.items() if isinstance(v, (int, float, str, bool))}


def train_model(model_name, ti):
    import db
    import train
    _batch(ti)
    splits = db.read_splits(db.get_engine())
    train.configure_mlflow()
    return train.train_candidate(model_name, splits, tags={"airflow_run_id": ti.run_id})


def select_and_promote(ti):
    import db
    import train
    _batch(ti)
    results = [ti.xcom_pull(task_ids=f"train_{m}") for m in MODELS]
    results = [r for r in results if r]
    if not results:
        raise ValueError("Ningún modelo terminó el entrenamiento.")
    splits = db.read_splits(db.get_engine())
    train.configure_mlflow()
    best = train.select_best(results)
    decision = train.register_and_promote(best, splits["test"])
    log.info("Decisión de promoción: %s", decision)
    return {k: v for k, v in decision.items() if isinstance(v, (int, float, str, bool, type(None)))}


# ---------------------------------------------------------------------------
# Definición del DAG
# ---------------------------------------------------------------------------
default_args = {
    "owner": "equipo-mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="diabetes_pipeline",
    description="Carga por lotes, limpieza, entrenamiento y promoción del champion",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=os.getenv("DAG_SCHEDULE", "@daily"),
    catchup=False,
    max_active_runs=1,       # un lote a la vez: evita lotes superpuestos
    is_paused_upon_creation=True,
    tags=["mlops", "diabetes"],
) as dag:
    t_init = PythonOperator(task_id="init_schema", python_callable=init_schema)
    t_check = PythonOperator(task_id="check_source", python_callable=check_source)
    t_plan = PythonOperator(task_id="plan_batch", python_callable=plan_batch)
    t_load = PythonOperator(task_id="load_raw", python_callable=load_raw)
    t_validate = PythonOperator(task_id="validate_raw", python_callable=validate_raw)
    t_clean = PythonOperator(task_id="preprocess", python_callable=preprocess)
    t_split = PythonOperator(task_id="assign_splits", python_callable=assign_splits)
    t_train = [
        PythonOperator(task_id=f"train_{m}", python_callable=train_model, op_kwargs={"model_name": m},
                       execution_timeout=timedelta(minutes=45))
        for m in MODELS
    ]
    t_promote = PythonOperator(task_id="select_and_promote", python_callable=select_and_promote)

    t_init >> t_check >> t_plan >> t_load >> t_validate >> t_clean >> t_split >> t_train >> t_promote
