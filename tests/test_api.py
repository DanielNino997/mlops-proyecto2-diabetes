"""
Pruebas de api/inference_api.py.

MLflow usa SQLite temporal (solo pruebas). Las pruebas de registro en la base
de datos necesitan TEST_DATABASE_URL (ver tests/test_db.py).

    python3 -m pytest -q tests/test_api.py
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sqlalchemy import text

import config
import db
import fake_data
import inference_api
import preprocessing
import schemas
import train

PATIENT = schemas.EXAMPLE_PATIENT.model_dump()


def metric_value(client, name, **labels):
    """Lee un valor del endpoint /metrics (0 si todavía no existe)."""
    total = 0.0
    for line in client.get("/metrics").text.splitlines():
        if line.startswith("#") or not line.startswith(name):
            continue
        head, _, value = line.rpartition(" ")
        series = head[len(name):]
        if all(f'{k}="{v}"' in series for k, v in labels.items()) and (
            series == "" or series.startswith("{")
        ):
            total += float(value)
    return total


@pytest.fixture(scope="module")
def splits():
    clean = preprocessing.clean_data(fake_data.make_fake_dataframe(5000, seed=31))
    return preprocessing.split_dataset(clean)


@pytest.fixture(scope="module")
def mlflow_uri(tmp_path_factory, splits):
    """Un MLflow temporal con un champion ya entrenado (v1)."""
    uri = f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'mlflow.db'}"
    train.configure_mlflow(uri, "api")
    train.train_all(splits, ["logistic_regression"])
    return uri


@pytest.fixture()
def client(mlflow_uri):
    store = inference_api.ModelStore(ttl_seconds=3600, tracking_uri=mlflow_uri)
    app = inference_api.create_app(store=store, use_default_engine=False)
    with TestClient(app) as test_client:
        test_client.store = store
        yield test_client


# ---------------------------------------------------------------------------
# Endpoints básicos
# ---------------------------------------------------------------------------
def test_health_ready_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    assert client.get("/ready").json()["model_version"] == "1"

    info = client.get("/model-info").json()
    assert info["status"] == "loaded"
    assert info["model_name"] == config.MODEL_NAME and info["model_alias"] == config.MODEL_ALIAS
    assert info["model_version"] == "1" and info["run_id"]
    assert 0 < info["threshold"] < 1
    assert info["features"] == schemas.FEATURE_COLUMNS
    assert info["refresh_interval_seconds"] == 3600


def test_predict_returns_every_required_field(client):
    response = client.post("/predict", json=PATIENT, headers={"X-Client": "pruebas"})
    assert response.status_code == 200
    body = response.json()

    # Campos mínimos que pide el enunciado.
    for field in ("prediction", "probability", "model_name", "model_version",
                  "model_alias", "processing_time_ms", "request_id"):
        assert field in body, field
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability"] <= 1
    assert body["model_version"] == "1" and body["model_alias"] == "champion"
    assert body["processing_time_ms"] > 0
    assert body["prediction"] == int(body["probability"] >= body["threshold"])
    assert body["prediction_label"]
    assert len({client.post("/predict", json=PATIENT).json()["request_id"] for _ in range(5)}) == 5


def test_a_higher_risk_patient_gets_a_higher_probability(client):
    low = {**PATIENT, "number_inpatient": 0, "number_emergency": 0, "insulin": "No"}
    high = {**PATIENT, "number_inpatient": 8, "number_emergency": 4, "insulin": "Up"}
    p_low = client.post("/predict", json=low).json()["probability"]
    p_high = client.post("/predict", json=high).json()["probability"]
    assert p_high > p_low


# ---------------------------------------------------------------------------
# Validación (lo que Streamlit muestra al usuario)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("change, field", [
    ({"age": "70"}, "age"),
    ({"race": "Martian"}, "race"),
    ({"time_in_hospital": 99}, "time_in_hospital"),
    ({"num_medications": -1}, "num_medications"),
    ({"diag_1": "428"}, "diag_1"),
    ({"campo_inventado": 1}, "campo_inventado"),
])
def test_invalid_input_is_rejected_with_details(client, change, field):
    response = client.post("/predict", json={**PATIENT, **change})
    assert response.status_code == 422
    assert field in [error["loc"][-1] for error in response.json()["detail"]]


def test_missing_fields_are_listed(client):
    broken = {k: v for k, v in PATIENT.items() if k not in ("age", "gender")}
    response = client.post("/predict", json=broken)
    assert response.status_code == 422
    missing = {error["loc"][-1] for error in response.json()["detail"]}
    assert {"age", "gender"} <= missing


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def test_metrics_endpoint_counts_requests_and_exposes_the_documented_names(client):
    before = metric_value(client, "diabetes_api_requests_total", endpoint="/predict", status="200")
    before_422 = metric_value(client, "diabetes_api_requests_total", endpoint="/predict", status="422")
    before_pred = metric_value(client, "diabetes_api_predictions_total")

    for _ in range(4):
        client.post("/predict", json=PATIENT)
    client.post("/predict", json={**PATIENT, "age": "x"})

    assert metric_value(client, "diabetes_api_requests_total", endpoint="/predict", status="200") == before + 4
    assert metric_value(client, "diabetes_api_requests_total", endpoint="/predict", status="422") == before_422 + 1
    assert metric_value(client, "diabetes_api_predictions_total") == before_pred + 4
    assert metric_value(client, "diabetes_api_request_duration_seconds_count", endpoint="/predict") >= 5

    text_metrics = client.get("/metrics").text
    for name in ("diabetes_api_requests_total", "diabetes_api_request_duration_seconds_bucket",
                 "diabetes_api_requests_in_progress", "diabetes_api_prediction_probability_bucket",
                 "diabetes_api_inference_duration_seconds_bucket", "diabetes_api_model_loaded",
                 "diabetes_api_model_version", "diabetes_api_model_info",
                 "process_cpu_seconds_total", "process_resident_memory_bytes"):
        assert name in text_metrics, name
    assert 'diabetes_api_model_info{model_alias="champion"' in text_metrics
    assert metric_value(client, "diabetes_api_model_loaded") == 1
    # /metrics no se cuenta a sí mismo.
    assert 'endpoint="/metrics"' not in text_metrics


def test_unknown_routes_do_not_explode_label_cardinality(client):
    client.get("/no-existe-1")
    client.get("/no-existe-2")
    assert metric_value(client, "diabetes_api_requests_total", endpoint="unmatched", status="404") >= 2
    assert "/no-existe-1" not in client.get("/metrics").text


# ---------------------------------------------------------------------------
# Carga del modelo desde MLflow
# ---------------------------------------------------------------------------
def test_without_a_champion_the_api_stays_up_but_cannot_predict(tmp_path):
    train.configure_mlflow(f"sqlite:///{tmp_path / 'vacio.db'}", "vacio")
    store = inference_api.ModelStore(ttl_seconds=3600, tracking_uri=f"sqlite:///{tmp_path / 'vacio.db'}")
    app = inference_api.create_app(store=store, use_default_engine=False)
    with TestClient(app) as empty:
        assert empty.get("/health").json() == {"status": "ok", "model_loaded": False}
        assert empty.get("/ready").status_code == 503
        info = empty.get("/model-info").json()
        assert info["status"] == "error" and info["model_version"] is None and info["last_error"]
        response = empty.post("/predict", json=PATIENT)
        assert response.status_code == 503
        assert metric_value(empty, "diabetes_api_prediction_errors_total", reason="model_unavailable") >= 1


def test_a_new_champion_is_used_without_restarting(tmp_path, splits, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'swap.db'}"
    train.configure_mlflow(uri, "swap")
    monkeypatch.setattr(train, "candidate_estimators",
                        lambda: {"weak": DummyClassifier(strategy="prior")})
    train.train_all(splits, ["weak"])                          # champion v1 (modelo débil)
    monkeypatch.undo()

    store = inference_api.ModelStore(ttl_seconds=3600, tracking_uri=uri)
    app = inference_api.create_app(store=store, use_default_engine=False)
    with TestClient(app) as swap:
        before = swap.post("/predict", json=PATIENT).json()
        assert before["model_version"] == "1"
        assert store.refresh() is False                       # nada cambió: no recarga

        train.configure_mlflow(uri, "swap")
        decision = train.train_all(splits, ["logistic_regression"])["decision"]
        assert decision["promoted"] and decision["version"] == 2

        assert store.refresh() is True                        # lo que hace el hilo cada TTL
        after = swap.post("/predict", json=PATIENT).json()
        assert after["model_version"] == "2"
        assert after["threshold"] != before["threshold"] or after["probability"] != before["probability"]
        assert swap.get("/model-info").json()["model_version"] == "2"
        assert metric_value(swap, "diabetes_api_model_version") == 2


def test_if_mlflow_goes_down_the_api_keeps_the_last_good_model(client):
    assert client.post("/predict", json=PATIENT).status_code == 200
    client.store.tracking_uri = "http://127.0.0.1:9"           # nadie escucha aquí
    assert client.store.refresh() is False
    assert client.store.last_error
    assert client.post("/predict", json=PATIENT).status_code == 200
    info = client.get("/model-info").json()
    assert info["status"] == "loaded" and info["model_version"] == "1" and info["last_error"]


def test_background_thread_refreshes_on_its_own(mlflow_uri):
    store = inference_api.ModelStore(ttl_seconds=1, tracking_uri=mlflow_uri)
    assert store.snapshot() is None
    store.start()
    try:
        import time
        deadline = time.time() + 20
        while store.snapshot() is None and time.time() < deadline:
            time.sleep(0.2)
        assert store.snapshot() is not None and store.snapshot().version == "1"
    finally:
        store.stop()


def test_concurrent_requests(client):
    with ThreadPoolExecutor(max_workers=16) as pool:
        codes = list(pool.map(lambda _: client.post("/predict", json=PATIENT).status_code, range(64)))
    assert set(codes) == {200}


# ---------------------------------------------------------------------------
# Registro de inferencias en PostgreSQL
# ---------------------------------------------------------------------------
needs_db = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"), reason="Define TEST_DATABASE_URL (ver tests/test_db.py)"
)


@pytest.fixture()
def db_client(mlflow_uri):
    engine = db.get_engine(os.environ["TEST_DATABASE_URL"])
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS inference_logs CASCADE"))
    db.init_schema(engine)
    store = inference_api.ModelStore(ttl_seconds=3600, tracking_uri=mlflow_uri)
    app = inference_api.create_app(store=store, engine=engine, use_default_engine=False)
    with TestClient(app) as test_client:
        test_client.engine = engine
        yield test_client
    engine.dispose()


@needs_db
def test_every_inference_is_logged_with_the_required_fields(db_client):
    response = db_client.post("/predict", json=PATIENT, headers={"X-Client": "locust"}).json()
    with db_client.engine.connect() as conn:
        rows = conn.execute(text("SELECT * FROM inference_logs")).mappings().all()
    assert len(rows) == 1
    row = rows[0]
    assert str(row["request_id"]) == response["request_id"]
    assert row["status"] == "success" and row["source"] == "locust"
    assert row["prediction"] == response["prediction"]
    assert row["probability"] == pytest.approx(response["probability"])
    assert row["model_name"] == config.MODEL_NAME and row["model_version"] == "1"
    assert row["model_alias"] == "champion"
    assert row["latency_ms"] == pytest.approx(response["processing_time_ms"], rel=0.01)
    assert row["created_at"] is not None
    assert row["input_data"] == json.loads(json.dumps(PATIENT))     # el JSON recibido, completo
    # El registro sirve para reentrenar: se puede reconstruir el DataFrame de entrada.
    assert set(row["input_data"]) == set(schemas.FEATURE_COLUMNS)


@needs_db
def test_invalid_requests_are_not_logged_and_counts_match(db_client):
    for _ in range(6):
        db_client.post("/predict", json=PATIENT)
    db_client.post("/predict", json={**PATIENT, "age": "x"})
    with db_client.engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM inference_logs")).scalar_one()
    assert count == 6


@needs_db
def test_a_failing_database_does_not_break_predictions(db_client):
    with db_client.engine.begin() as conn:
        conn.execute(text("DROP TABLE inference_logs"))
    before = metric_value(db_client, "diabetes_api_inference_log_writes_total", result="error")
    response = db_client.post("/predict", json=PATIENT)
    assert response.status_code == 200                                # el usuario recibe su predicción
    assert metric_value(db_client, "diabetes_api_inference_log_writes_total", result="error") == before + 1
