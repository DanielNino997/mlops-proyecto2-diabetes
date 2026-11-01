"""
Pruebas de src/db.py contra un PostgreSQL real.

Se saltan solas si no hay base de datos de prueba. Para correrlas:

    export TEST_DATABASE_URL=postgresql+psycopg2://usuario:clave@localhost:5432/diabetes_data
    python3 -m pytest -q tests/test_db.py

OJO: estas pruebas BORRAN las tablas raw_data, clean_data, ingestion_batches e
inference_logs de esa base. Usa una base de pruebas, nunca la del clúster.
"""

import os

import pandas as pd
import pytest
from sqlalchemy import text

import config
import db
import fake_data
import preprocessing

pytestmark = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="Define TEST_DATABASE_URL para correr las pruebas con PostgreSQL",
)

N_ROWS = 4000
BATCH = 1500


@pytest.fixture()
def engine():
    eng = db.get_engine(os.environ["TEST_DATABASE_URL"])
    with eng.begin() as conn:
        for table in ["inference_logs", "clean_data", "raw_data", "ingestion_batches"]:
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
    db.init_schema(eng)
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def source() -> pd.DataFrame:
    return fake_data.make_fake_dataframe(N_ROWS, seed=3)


def run_batch(engine, source, run_id, batch_size=BATCH):
    """Ejecuta las tareas de un lote en el mismo orden que el DAG."""
    batch = db.plan_batch(engine, run_id, "fake.csv", len(source), batch_size)
    if batch is None:
        return None
    load = db.load_raw_batch(engine, batch, source)
    valid = db.validate_batch(engine, batch["batch_id"])
    clean = db.process_batch(engine, batch["batch_id"])
    splits = db.assign_splits(engine)
    return batch, load, valid, clean, splits


def scalar(engine, sql, **params):
    with engine.connect() as conn:
        return conn.execute(text(sql), params).scalar_one()


# ---------------------------------------------------------------------------
# Esquema
# ---------------------------------------------------------------------------
def test_init_schema_is_idempotent_and_matches_contract(engine):
    db.init_schema(engine)  # segunda vez: no debe fallar
    db.init_schema(engine)

    def columns(table):
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = :t ORDER BY ordinal_position"), {"t": table})
            return [r[0] for r in rows]

    # raw_data (esquema de Ana): cada fila completa va en payload JSONB
    raw_cols = columns("raw_data")
    assert {"batch_id", "load_ts", "source_file", "row_hash", "status", "payload"} <= set(raw_cols)

    # clean_data: columnas del contrato + las de control + split (lo agrega pipeline.sql)
    assert set(preprocessing.CLEAN_COLUMNS) <= set(columns("clean_data"))
    assert {"batch_id", "row_hash", "split", "processed_at"} <= set(columns("clean_data"))
    assert "request_id" in columns("inference_logs")
    assert set(columns("ingestion_batches")) >= {"batch_id", "dag_run_id", "row_start", "row_end", "status"}


# ---------------------------------------------------------------------------
# Archivo fuente
# ---------------------------------------------------------------------------
def test_check_source_file(tmp_path, source):
    good = tmp_path / "ok.csv"
    source.to_csv(good, index=False)
    info = db.check_source_file(str(good))
    assert info["rows"] == N_ROWS and info["columns"] == 50

    with pytest.raises(FileNotFoundError):
        db.check_source_file(str(tmp_path / "no_existe.csv"))

    empty = tmp_path / "vacio.csv"
    empty.write_text("")
    with pytest.raises(ValueError):
        db.check_source_file(str(empty))

    broken = tmp_path / "roto.csv"
    source.drop(columns=["age"]).to_csv(broken, index=False)
    with pytest.raises(ValueError, match="faltan columnas"):
        db.check_source_file(str(broken))


# ---------------------------------------------------------------------------
# Lotes
# ---------------------------------------------------------------------------
def test_batches_cover_the_file_without_overlap(engine, source):
    seen = 0
    for i in range(1, 10):
        result = run_batch(engine, source, f"run-{i}")
        if result is None:
            break
        batch = result[0]
        assert batch["row_start"] == seen
        assert batch["row_end"] - batch["row_start"] <= config.BATCH_SIZE
        seen = batch["row_end"]
    assert seen == N_ROWS
    assert scalar(engine, "SELECT COUNT(*) FROM ingestion_batches") == 3  # 1500+1500+1000
    assert scalar(engine, "SELECT COUNT(*) FROM raw_data") == N_ROWS
    assert scalar(engine, "SELECT COUNT(*) FROM ingestion_batches WHERE status <> 'processed'") == 0
    # Con el archivo agotado, otra ejecución no carga nada.
    assert db.plan_batch(engine, "run-extra", "fake.csv", N_ROWS, BATCH) is None


def test_batch_size_limit_is_enforced(engine, source):
    with pytest.raises(ValueError):
        db.plan_batch(engine, "run-x", "fake.csv", N_ROWS, config.BATCH_SIZE + 1)


def test_retry_in_same_run_reuses_batch_and_does_not_duplicate(engine, source):
    first = db.plan_batch(engine, "run-1", "fake.csv", N_ROWS, BATCH)
    db.load_raw_batch(engine, first, source)

    again = db.plan_batch(engine, "run-1", "fake.csv", N_ROWS, BATCH)  # reintento
    assert again["batch_id"] == first["batch_id"] and again["resumed"] is True
    report = db.load_raw_batch(engine, again, source)  # recarga el mismo lote
    assert report["inserted"] == 0 and report["ignored_duplicates"] == BATCH
    assert scalar(engine, "SELECT COUNT(*) FROM raw_data") == BATCH


def test_next_run_resumes_unfinished_batch(engine, source):
    first = db.plan_batch(engine, "run-1", "fake.csv", N_ROWS, BATCH)
    db.load_raw_batch(engine, first, source)  # la ejecución 1 muere aquí

    second = db.plan_batch(engine, "run-2", "fake.csv", N_ROWS, BATCH)
    assert second["batch_id"] == first["batch_id"]  # no se salta el lote 1
    assert second["resumed"] is True
    assert scalar(engine, "SELECT COUNT(*) FROM ingestion_batches") == 1


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------
def test_validation_rejects_bad_rows_but_keeps_them(engine, source):
    dirty = source.copy()
    dirty.loc[10:19, "readmitted"] = "quizas"       # 10 filas con target inválido
    dirty.loc[30:34, "patient_nbr"] = "abc"          # 5 con paciente inválido
    batch = db.plan_batch(engine, "run-1", "fake.csv", len(dirty), BATCH)
    db.load_raw_batch(engine, batch, dirty)
    report = db.validate_batch(engine, batch["batch_id"])

    assert report["by_status"]["rejected"] == 15
    assert report["by_status"]["validated"] == BATCH - 15
    assert scalar(engine, "SELECT COUNT(*) FROM raw_data WHERE status = 'rejected'") == 15

    db.process_batch(engine, batch["batch_id"])
    rejected_ids = set(dirty.loc[10:19, "encounter_id"]) | set(dirty.loc[30:34, "encounter_id"])
    with engine.connect() as conn:
        in_clean = {r[0] for r in conn.execute(text("SELECT encounter_id FROM clean_data"))}
    assert not (in_clean & {int(i) for i in rejected_ids})


def test_validation_fails_when_too_many_rows_are_bad(engine, source):
    dirty = source.copy()
    dirty.loc[0:299, "readmitted"] = "mal"            # 20% del lote
    batch = db.plan_batch(engine, "run-1", "fake.csv", len(dirty), BATCH)
    db.load_raw_batch(engine, batch, dirty)
    with pytest.raises(ValueError, match="Calidad insuficiente"):
        db.validate_batch(engine, batch["batch_id"])
    assert scalar(engine, "SELECT status FROM ingestion_batches WHERE batch_id = :b",
                  b=batch["batch_id"]) == "failed"
    assert scalar(engine, "SELECT COUNT(*) FROM clean_data") == 0


# ---------------------------------------------------------------------------
# Limpieza y split
# ---------------------------------------------------------------------------
def test_process_matches_direct_cleaning_and_is_idempotent(engine, source):
    batch, load, valid, clean, splits = run_batch(engine, source, "run-1")
    expected = preprocessing.clean_data(source.iloc[:BATCH])

    assert clean["filas_salida"] == len(expected)
    stored = db.read_clean(engine)
    assert sorted(stored["encounter_id"]) == sorted(expected["encounter_id"])

    merged = stored.merge(expected, on="encounter_id", suffixes=("_db", ""))
    for column in preprocessing.CLEAN_COLUMNS[1:]:
        assert (merged[column + "_db"] == merged[column]).all(), column

    # Estados de raw_data: todo lo válido terminó 'processed' o 'excluded'.
    by_status = dict(engine.connect().execute(text(
        "SELECT status, COUNT(*) FROM raw_data GROUP BY status")).all())
    assert by_status.get("processed") == len(expected)
    assert sum(by_status.values()) == BATCH

    # Reprocesar no duplica filas.
    db.process_batch(engine, batch["batch_id"])
    assert scalar(engine, "SELECT COUNT(*) FROM clean_data") == len(expected)


def test_splits_never_share_a_patient_across_batches(engine, source):
    while run_batch(engine, source, f"run-{scalar(engine, 'SELECT COUNT(*) FROM ingestion_batches') + 1}"):
        pass
    summary = db.split_summary(engine)
    assert summary["patients_in_more_than_one_split"] == 0
    assert "sin_asignar" not in summary["splits"]

    frame = db.read_clean(engine)
    assert frame.groupby("patient_nbr")["split"].nunique().max() == 1
    # El split guardado coincide con la función determinista.
    assert (frame["patient_nbr"].map(preprocessing.split_of) == frame["split"]).all()

    parts = db.read_splits(engine)
    assert sum(len(p) for p in parts.values()) == len(frame)
    assert all(p[config.TARGET_COLUMN].sum() > 0 for p in parts.values())


def test_assign_splits_is_idempotent(engine, source):
    run_batch(engine, source, "run-1")
    before = db.read_clean(engine)[["encounter_id", "split"]]
    again = db.assign_splits(engine)
    assert again["newly_assigned_patients"] == 0
    after = db.read_clean(engine)[["encounter_id", "split"]]
    pd.testing.assert_frame_equal(before, after)
