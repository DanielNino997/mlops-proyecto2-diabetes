"""
Acceso a PostgreSQL para el DAG: carga por lotes, validación, limpieza y split.

Cada función hace una sola cosa y se puede reintentar sin duplicar datos
(idempotente). Las tareas del DAG (airflow/dags/diabetes_pipeline.py) solo
llaman a estas funciones; así la lógica se puede probar sin Airflow.

Ciclo de vida de una fila de raw_data (columna status):

    loaded -> validated -> processed
                  |             (está en clean_data)
                  +--------> excluded   (válida, pero la limpieza la descarta:
                  |                      fallecido/hospicio o género inválido)
    loaded -> rejected                  (falló la validación de calidad)

Ciclo de vida de un lote (ingestion_batches.status):

    planned -> loaded -> validated -> processed
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from psycopg2.extras import Json, execute_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

import config
import preprocessing
import schemas

logger = logging.getLogger(__name__)

# Si más de este porcentaje de un lote falla la validación, el lote completo
# se rechaza y la tarea falla (en vez de entrenar con datos dudosos).
MAX_REJECTED_RATIO = float(os.getenv("MAX_REJECTED_RATIO", "0.05"))

_BATCH_LOCK_KEY = 20262910  # clave del lock que serializa la planeación de lotes
_INSERT_PAGE_SIZE = 2000


# ---------------------------------------------------------------------------
# Conexión y esquema
# ---------------------------------------------------------------------------
def get_engine(url: Optional[str] = None) -> Engine:
    """Engine de SQLAlchemy; con pool_pre_ping para sobrevivir a reinicios de Postgres."""
    return create_engine(url or config.postgres_url(), pool_pre_ping=True, future=True)


def _find_sql(name: str) -> Path:
    """Busca sql/<name>: junto al repo, en la imagen o en SQL_DIR."""
    candidates = [
        Path(os.getenv("SQL_DIR", "/nonexistent")) / name,
        Path(__file__).resolve().parents[1] / "sql" / name,
        Path("/opt/airflow/sql") / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No se encontró sql/{name}. Define SQL_DIR o copia la carpeta sql/ a la "
        "imagen en /opt/airflow/sql/."
    )


def init_schema(engine: Engine) -> None:
    """
    Crea las tablas si no existen (puede correr varias veces):
    init.sql (raw_data, clean_data, inference_logs) y luego pipeline.sql
    (ingestion_batches y la columna clean_data.split).
    """
    for name in ("init.sql", "pipeline.sql"):
        path = _find_sql(name)
        with engine.begin() as conn:
            conn.exec_driver_sql(path.read_text(encoding="utf-8"))
        logger.info("Esquema verificado con %s", path)


def _quote(column: str) -> str:
    return '"%s"' % column.replace('"', '""')


# ---------------------------------------------------------------------------
# Archivo fuente
# ---------------------------------------------------------------------------
def read_source_file(path: str) -> pd.DataFrame:
    """Lee el CSV completo como texto, sin convertir nada (ni "?" ni "None")."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def check_source_file(path: str) -> Dict[str, object]:
    """
    Verifica que el archivo fuente exista, no esté vacío y tenga las columnas
    esperadas. Devuelve cuántas filas tiene.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"No existe el archivo fuente: {path}")
    if file.stat().st_size == 0:
        raise ValueError(f"El archivo fuente está vacío: {path}")

    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in config.SOURCE_COLUMNS if c not in header]
    extra = [c for c in header if c not in config.SOURCE_COLUMNS]
    if missing:
        raise ValueError(f"Al archivo le faltan columnas: {missing}")
    if extra:
        logger.warning("El archivo trae columnas que se ignoran: %s", extra)

    rows = len(pd.read_csv(path, usecols=[0], dtype=str, keep_default_na=False))
    if rows == 0:
        raise ValueError("El archivo no tiene filas de datos")
    return {"path": str(path), "bytes": file.stat().st_size, "rows": rows,
            "columns": len(header)}


def compute_row_hash(values: List[str]) -> str:
    """SHA-256 del contenido de la fila; identifica la fila sin importar el lote."""
    return hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Planeación de lotes
# ---------------------------------------------------------------------------
def plan_batch(
    engine: Engine,
    dag_run_id: str,
    source_file: str,
    total_rows: int,
    batch_size: int = config.BATCH_SIZE,
) -> Optional[Dict[str, object]]:
    """
    Decide qué lote le toca a esta ejecución del DAG.

    1. Si esta ejecución ya tiene un lote (reintento de la tarea), lo reutiliza.
    2. Si quedó un lote incompleto de una ejecución anterior que falló, lo
       retoma en vez de saltárselo.
    3. Si no, crea el siguiente lote del archivo.

    Devuelve None cuando el archivo ya se cargó completo.
    """
    if batch_size <= 0 or batch_size > config.BATCH_SIZE:
        raise ValueError(f"batch_size debe estar entre 1 y {config.BATCH_SIZE}")

    with engine.begin() as conn:
        # Serializa la planeación: dos ejecuciones simultáneas no pueden
        # reclamar el mismo lote.
        conn.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": _BATCH_LOCK_KEY})

        own = conn.execute(
            text("SELECT * FROM ingestion_batches WHERE dag_run_id = :r"),
            {"r": dag_run_id},
        ).mappings().first()
        if own:
            return {**dict(own), "resumed": True, "total_rows": total_rows}

        pending = conn.execute(
            text("SELECT * FROM ingestion_batches WHERE status <> 'processed' "
                 "ORDER BY batch_id LIMIT 1")
        ).mappings().first()
        if pending:
            conn.execute(
                text("UPDATE ingestion_batches SET dag_run_id = :r, status = "
                     "CASE WHEN status = 'failed' THEN 'planned' ELSE status END, "
                     "error_message = NULL, updated_at = now() WHERE batch_id = :b"),
                {"r": dag_run_id, "b": pending["batch_id"]},
            )
            batch = conn.execute(
                text("SELECT * FROM ingestion_batches WHERE batch_id = :b"),
                {"b": pending["batch_id"]},
            ).mappings().first()
            logger.warning("Se retoma el lote %s que quedó incompleto", batch["batch_id"])
            return {**dict(batch), "resumed": True, "total_rows": total_rows}

        last = conn.execute(
            text("SELECT COALESCE(MAX(row_end), 0) AS row_end, "
                 "COALESCE(MAX(batch_id), 0) AS batch_id FROM ingestion_batches")
        ).mappings().first()
        start = int(last["row_end"])
        if start >= total_rows:
            logger.info("El archivo ya se cargó completo (%s filas)", total_rows)
            return None

        end = min(start + batch_size, total_rows)
        batch_id = int(last["batch_id"]) + 1
        conn.execute(
            text("INSERT INTO ingestion_batches (batch_id, dag_run_id, source_file, "
                 "row_start, row_end) VALUES (:b, :r, :f, :s, :e)"),
            {"b": batch_id, "r": dag_run_id, "f": source_file, "s": start, "e": end},
        )
        return {
            "batch_id": batch_id, "dag_run_id": dag_run_id, "source_file": source_file,
            "row_start": start, "row_end": end, "status": "planned",
            "resumed": False, "total_rows": total_rows,
        }


def set_batch_status(
    engine: Engine, batch_id: int, status: str, error: Optional[str] = None,
    loaded_rows: Optional[int] = None,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE ingestion_batches SET status = :s, error_message = :e, "
                 "loaded_rows = COALESCE(:n, loaded_rows), updated_at = now() "
                 "WHERE batch_id = :b"),
            {"s": status, "e": error, "n": loaded_rows, "b": batch_id},
        )


# ---------------------------------------------------------------------------
# Carga a raw_data
# ---------------------------------------------------------------------------
def load_raw_batch(engine: Engine, batch: Dict[str, object], source: pd.DataFrame) -> Dict[str, int]:
    """
    Inserta en raw_data las filas del lote tal como vienen del archivo.

    Es idempotente: row_hash es único, así que si el lote ya estaba cargado
    (total o parcialmente) las filas repetidas se ignoran.
    """
    batch_id = int(batch["batch_id"])
    start, end = int(batch["row_start"]), int(batch["row_end"])
    chunk = source.iloc[start:end][config.SOURCE_COLUMNS]
    if chunk.empty:
        raise ValueError(f"El lote {batch_id} no tiene filas (rango {start}-{end})")

    # raw_data guarda cada fila completa, sin transformar, en la columna payload (JSONB).
    rows = []
    for values in chunk.itertuples(index=False, name=None):
        values = [str(v) for v in values]
        payload = dict(zip(config.SOURCE_COLUMNS, values))
        rows.append((batch_id, str(batch["source_file"]), compute_row_hash(values), Json(payload)))

    sql = (
        f"INSERT INTO {config.RAW_TABLE} (batch_id, source_file, row_hash, payload) "
        "VALUES %s ON CONFLICT (row_hash) DO NOTHING RETURNING 1"
    )
    inserted = 0
    with engine.begin() as conn:
        cursor = conn.connection.cursor()
        returned = execute_values(cursor, sql, rows, page_size=_INSERT_PAGE_SIZE, fetch=True)
        inserted = len(returned)
        cursor.close()

        total_in_batch = conn.execute(
            text(f"SELECT COUNT(*) FROM {config.RAW_TABLE} WHERE batch_id = :b"),
            {"b": batch_id},
        ).scalar_one()
        conn.execute(
            text("UPDATE ingestion_batches SET loaded_rows = :n, updated_at = now(), "
                 "status = CASE WHEN status IN ('planned', 'failed') THEN 'loaded' "
                 "ELSE status END, error_message = NULL WHERE batch_id = :b"),
            {"n": total_in_batch, "b": batch_id},
        )

    report = {
        "batch_id": batch_id,
        "rows_in_batch": len(rows),
        "inserted": inserted,
        "ignored_duplicates": len(rows) - inserted,
        "rows_in_raw_for_batch": int(total_in_batch),
    }
    logger.info("Carga del lote %s: %s", batch_id, report)
    return report


# ---------------------------------------------------------------------------
# Validación de calidad
# ---------------------------------------------------------------------------
_VALID_ROW_SQL = """
    COALESCE(payload->>'encounter_id' ~ '^[0-9]+$', false)
    AND COALESCE(payload->>'patient_nbr' ~ '^[0-9]+$', false)
    AND COALESCE(payload->>'readmitted' IN ('NO', '>30', '<30'), false)
"""


def validate_batch(engine: Engine, batch_id: int) -> Dict[str, object]:
    """
    Validación básica de calidad del lote en raw_data:

      * el lote no está vacío;
      * encounter_id y patient_nbr son números;
      * readmitted trae uno de los tres valores conocidos;
      * no hay encuentros repetidos dentro del lote.

    Las filas que fallan quedan como 'rejected' (se conservan para auditoría).
    Si fallan más del MAX_REJECTED_RATIO, la tarea lanza error.
    """
    with engine.begin() as conn:
        total = conn.execute(
            text(f"SELECT COUNT(*) FROM {config.RAW_TABLE} WHERE batch_id = :b"),
            {"b": batch_id},
        ).scalar_one()
        if total == 0:
            raise ValueError(f"El lote {batch_id} no tiene filas en raw_data")

        duplicated = conn.execute(
            text(f"SELECT COUNT(*) - COUNT(DISTINCT payload->>'encounter_id') FROM {config.RAW_TABLE} "
                 "WHERE batch_id = :b"),
            {"b": batch_id},
        ).scalar_one()

        rejected = conn.execute(
            text(f"UPDATE {config.RAW_TABLE} SET status = 'rejected' "
                 f"WHERE batch_id = :b AND status = 'loaded' AND NOT ({_VALID_ROW_SQL})"),
            {"b": batch_id},
        ).rowcount
        validated = conn.execute(
            text(f"UPDATE {config.RAW_TABLE} SET status = 'validated' "
                 "WHERE batch_id = :b AND status = 'loaded'"),
            {"b": batch_id},
        ).rowcount

        counts = dict(conn.execute(
            text(f"SELECT status, COUNT(*) FROM {config.RAW_TABLE} WHERE batch_id = :b "
                 "GROUP BY status"),
            {"b": batch_id},
        ).all())

        unknown_marks = conn.execute(
            text(f"""SELECT
                  COUNT(*) FILTER (WHERE payload->>'race' = '?') AS race_unknown,
                  COUNT(*) FILTER (WHERE payload->>'weight' = '?') AS weight_unknown,
                  COUNT(*) FILTER (WHERE payload->>'diag_1' = '?') AS diag_1_unknown
                FROM {config.RAW_TABLE} WHERE batch_id = :b"""),
            {"b": batch_id},
        ).mappings().one()

    n_rejected = int(counts.get("rejected", 0))
    ratio = n_rejected / total
    report = {
        "batch_id": batch_id,
        "rows": int(total),
        "newly_validated": int(validated),
        "newly_rejected": int(rejected),
        "by_status": {k: int(v) for k, v in counts.items()},
        "rejected_ratio": round(ratio, 5),
        "duplicated_encounters": int(duplicated),
        "unknown_marks": {k: int(v) for k, v in unknown_marks.items()},
    }

    problems = []
    if ratio > MAX_REJECTED_RATIO:
        problems.append(f"{ratio:.1%} de filas rechazadas (máximo {MAX_REJECTED_RATIO:.0%})")
    if duplicated:
        problems.append(f"{duplicated} encuentros repetidos dentro del lote")
    if problems:
        message = "; ".join(problems)
        set_batch_status(engine, batch_id, "failed", error=message)
        raise ValueError(f"Calidad insuficiente en el lote {batch_id}: {message}")

    set_batch_status(engine, batch_id, "validated")
    logger.info("Validación del lote %s: %s", batch_id, report)
    return report


# ---------------------------------------------------------------------------
# Limpieza: raw_data -> clean_data
# ---------------------------------------------------------------------------
def read_raw_batch(engine: Engine, batch_id: int, status: str) -> pd.DataFrame:
    """Lee las filas de un lote con cierto estado; NULL se devuelve como vacío."""
    with engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT row_hash, payload FROM {config.RAW_TABLE} "
                 "WHERE batch_id = :b AND status = :s ORDER BY id"),
            {"b": batch_id, "s": status},
        ).all()
    records = []
    for row_hash, payload in result:
        data = payload if isinstance(payload, dict) else json.loads(payload)
        records.append({"row_hash": row_hash, **{c: data.get(c, "") for c in config.SOURCE_COLUMNS}})
    return pd.DataFrame(records, columns=["row_hash"] + config.SOURCE_COLUMNS).fillna("")


def _to_python_rows(frame: pd.DataFrame) -> List[tuple]:
    """DataFrame -> tuplas con tipos nativos de Python (psycopg2 no entiende numpy)."""
    as_object = frame.astype(object).where(frame.notna(), None)
    return [tuple(row) for row in as_object.itertuples(index=False, name=None)]


def process_batch(engine: Engine, batch_id: int) -> Dict[str, object]:
    """
    Limpia las filas 'validated' del lote y las guarda en clean_data.

    El resultado es idempotente: clean_data usa encounter_id como llave y
    ON CONFLICT DO UPDATE, así que reprocesar un lote no duplica filas.
    """
    raw = read_raw_batch(engine, batch_id, "validated")
    report: Dict[str, object] = {"batch_id": batch_id, "rows_to_process": len(raw)}

    if raw.empty:
        report.update(filas_salida=0, note="no había filas validadas (lote ya procesado)")
        set_batch_status(engine, batch_id, "processed")
        return report

    hashes = dict(zip(pd.to_numeric(raw["encounter_id"]).astype("int64"), raw["row_hash"]))
    clean_report: dict = {}
    clean = preprocessing.clean_data(raw.drop(columns=["row_hash"]), clean_report)

    invalid, samples = preprocessing.validate_against_contract(clean)
    if invalid:
        raise ValueError(
            f"{invalid} filas limpias no cumplen schemas.PatientFeatures: {samples}"
        )

    clean = clean.copy()
    clean["batch_id"] = batch_id
    clean["row_hash"] = clean["encounter_id"].map(hashes)

    columns = list(clean.columns)
    updates = ", ".join(
        f"{_quote(c)} = EXCLUDED.{_quote(c)}" for c in columns if c != "encounter_id"
    )
    sql = (
        f"INSERT INTO {config.CLEAN_TABLE} ({', '.join(_quote(c) for c in columns)}) "
        f"VALUES %s ON CONFLICT (encounter_id) DO UPDATE SET {updates}, processed_at = now()"
    )

    kept_hashes = list(clean["row_hash"])
    with engine.begin() as conn:
        cursor = conn.connection.cursor()
        execute_values(cursor, sql, _to_python_rows(clean), page_size=_INSERT_PAGE_SIZE)
        cursor.close()

        processed = conn.execute(
            text(f"UPDATE {config.RAW_TABLE} SET status = 'processed' "
                 "WHERE batch_id = :b AND status = 'validated' AND row_hash = ANY(:h)"),
            {"b": batch_id, "h": kept_hashes},
        ).rowcount
        excluded = conn.execute(
            text(f"UPDATE {config.RAW_TABLE} SET status = 'excluded' "
                 "WHERE batch_id = :b AND status = 'validated'"),
            {"b": batch_id},
        ).rowcount

    set_batch_status(engine, batch_id, "processed")
    report.update(clean_report)
    report.update(raw_marked_processed=processed, raw_marked_excluded=excluded)
    logger.info("Limpieza del lote %s: %s", batch_id, report)
    return report


# ---------------------------------------------------------------------------
# Split train / val / test
# ---------------------------------------------------------------------------
def assign_splits(engine: Engine) -> Dict[str, object]:
    """
    Asigna train/val/test a los pacientes de clean_data que aún no lo tienen.

    La asignación depende solo del patient_nbr (hash), así que un paciente que
    reaparece en un lote posterior cae siempre en el mismo conjunto.
    """
    with engine.begin() as conn:
        patients = [
            row[0] for row in conn.execute(
                text(f"SELECT DISTINCT patient_nbr FROM {config.CLEAN_TABLE} WHERE split IS NULL")
            )
        ]
        if patients:
            values = [(int(p), preprocessing.split_of(p)) for p in patients]
            cursor = conn.connection.cursor()
            execute_values(
                cursor,
                f"UPDATE {config.CLEAN_TABLE} AS c SET split = v.split "
                "FROM (VALUES %s) AS v(patient_nbr, split) "
                "WHERE c.patient_nbr = v.patient_nbr::bigint",
                values,
                page_size=_INSERT_PAGE_SIZE,
            )
            cursor.close()

    return split_summary(engine, newly_assigned_patients=len(patients))


def split_summary(engine: Engine, newly_assigned_patients: int = 0) -> Dict[str, object]:
    """Filas, pacientes y % de positivos por conjunto, y verificación de fuga."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"""SELECT split, COUNT(*) AS rows, COUNT(DISTINCT patient_nbr) AS patients,
                            SUM(readmitted_lt30) AS positives
                     FROM {config.CLEAN_TABLE} GROUP BY split ORDER BY split""")
        ).mappings().all()
        leaked = conn.execute(
            text(f"""SELECT COUNT(*) FROM (
                       SELECT patient_nbr FROM {config.CLEAN_TABLE}
                       GROUP BY patient_nbr HAVING COUNT(DISTINCT split) > 1) t""")
        ).scalar_one()

    summary = {
        "newly_assigned_patients": newly_assigned_patients,
        "patients_in_more_than_one_split": int(leaked),
        "splits": {},
    }
    for row in rows:
        name = row["split"] or "sin_asignar"
        n = int(row["rows"])
        summary["splits"][name] = {
            "rows": n,
            "patients": int(row["patients"]),
            "positives": int(row["positives"]),
            "positive_rate": round(int(row["positives"]) / n, 4) if n else 0.0,
        }
    if leaked:
        raise ValueError(f"{leaked} pacientes están en más de un conjunto")
    return summary


def read_clean(engine: Engine, split: Optional[str] = None) -> pd.DataFrame:
    """Lee clean_data (todas las columnas del contrato + split)."""
    columns = preprocessing.CLEAN_COLUMNS + ["split"]
    sql = f"SELECT {', '.join(_quote(c) for c in columns)} FROM {config.CLEAN_TABLE}"
    params: Dict[str, object] = {}
    if split:
        sql += " WHERE split = :s"
        params["s"] = split
    sql += " ORDER BY encounter_id"
    # Sin pd.read_sql_query: pandas 2.2+ exige SQLAlchemy 2, y Airflow 3.0 trae la 1.4.
    with engine.connect() as conn:
        result = conn.execute(text(sql), params)
        rows = result.fetchall()
    return pd.DataFrame(rows, columns=columns)


def read_splits(engine: Engine) -> Dict[str, pd.DataFrame]:
    """Los tres conjuntos de entrenamiento, validación y prueba."""
    frame = read_clean(engine)
    if frame["split"].isna().any():
        raise ValueError("Hay filas de clean_data sin split; falta correr assign_splits")
    return {
        name: frame[frame["split"] == name].drop(columns=["split"]).reset_index(drop=True)
        for name in preprocessing.SPLIT_NAMES
    }


# Lista de columnas que el contrato exige en clean_data, útil para pruebas.
CLEAN_TABLE_COLUMNS = preprocessing.CLEAN_COLUMNS + ["batch_id", "row_hash", "split", "processed_at"]
FEATURES = schemas.FEATURE_COLUMNS
