"""
API de inferencia (FastAPI).

Endpoints:
    GET  /health      la API está viva (liveness)
    GET  /ready       la API tiene un modelo cargado (readiness)
    GET  /model-info  nombre, versión y estado del modelo en uso
    POST /predict     recibe un paciente y devuelve la predicción
    GET  /metrics     métricas en formato Prometheus

Estrategia de carga del modelo (la pide el enunciado explicada):

  * Al iniciar, la API consulta a MLflow qué versión tiene el alias
    "champion" y descarga ESA versión. No hay ninguna ruta de archivo en el
    código: el modelo siempre viene de MLflow.
  * Un hilo en segundo plano repite esa consulta cada MODEL_CACHE_TTL_SECONDS.
    Solo descarga el modelo si la versión cambió; si no, no hace nada. Así un
    champion nuevo se usa sin reiniciar ni tocar código.
  * /predict nunca consulta MLflow: usa el modelo que ya está en memoria. La
    latencia de una predicción no depende de MLflow ni de MinIO.
  * Si MLflow no responde, la API sigue sirviendo con el último modelo bueno.

Cada predicción se guarda en la tabla inference_logs, después de enviar la
respuesta (BackgroundTasks), para que escribir en la base de datos no sume
latencia. Si la escritura falla, se cuenta en una métrica y la predicción no
se pierde para el usuario.

Despliegue: un solo proceso por pod. Para más capacidad se suben réplicas del
Deployment (las métricas de Prometheus de cada pod se suman en Grafana).
"""

import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# En el repositorio, config.py y schemas.py viven en ../src; en la imagen de
# Docker se copian junto a este archivo.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

# MLflow reintenta muchas veces por defecto; con el servidor caído eso dejaría
# colgado el arranque. Valores más cortos, sin pisar lo que se defina afuera.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "15")

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Response  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from prometheus_client import (  # noqa: E402
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.engine import Engine  # noqa: E402

import config  # noqa: E402
import schemas  # noqa: E402

logger = logging.getLogger("inference_api")

POSITIVE_LABEL = "Readmisión en menos de 30 días"
NEGATIVE_LABEL = "Sin readmisión en menos de 30 días"

# Cuánto esperar entre intentos de carga cuando todavía no hay modelo.
RETRY_SECONDS_WITHOUT_MODEL = 10


# ---------------------------------------------------------------------------
# Métricas de Prometheus (los nombres están documentados en docs/metricas.md)
# ---------------------------------------------------------------------------
REQUESTS = Counter(
    "diabetes_api_requests_total",
    "Solicitudes HTTP atendidas por la API",
    ["method", "endpoint", "status"],
)
LATENCY = Histogram(
    "diabetes_api_request_duration_seconds",
    "Duración de las solicitudes HTTP, en segundos",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
IN_PROGRESS = Gauge(
    "diabetes_api_requests_in_progress", "Solicitudes que se están atendiendo ahora mismo"
)
PREDICTIONS = Counter(
    "diabetes_api_predictions_total", "Predicciones entregadas, por clase", ["prediction"]
)
PROBABILITY = Histogram(
    "diabetes_api_prediction_probability",
    "Distribución de la probabilidad predicha (sirve para vigilar deriva)",
    buckets=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
INFERENCE_SECONDS = Histogram(
    "diabetes_api_inference_duration_seconds",
    "Tiempo de la inferencia del modelo (sin red ni base de datos)",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
PREDICTION_ERRORS = Counter(
    "diabetes_api_prediction_errors_total", "Errores al predecir, por causa", ["reason"]
)
LOG_WRITES = Counter(
    "diabetes_api_inference_log_writes_total",
    "Escrituras a inference_logs, por resultado",
    ["result"],
)
MODEL_LOADED = Gauge("diabetes_api_model_loaded", "1 si hay un modelo cargado en memoria")
MODEL_VERSION = Gauge("diabetes_api_model_version", "Número de versión del modelo en uso")
MODEL_INFO = Gauge(
    "diabetes_api_model_info", "Modelo en uso (valor 1; los datos van en las etiquetas)",
    ["model_name", "model_version", "model_alias"],
)
MODEL_LOADED_AT = Gauge(
    "diabetes_api_model_loaded_timestamp_seconds", "Momento (epoch) en que se cargó el modelo"
)
MODEL_RELOADS = Counter(
    "diabetes_api_model_reloads_total", "Cargas de modelo, por resultado", ["result"]
)


# ---------------------------------------------------------------------------
# Modelo en memoria
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadedModel:
    """Foto inmutable del modelo en uso; se reemplaza completa al recargar."""

    model: Any
    name: str
    version: str
    alias: str
    run_id: Optional[str]
    threshold: float
    loaded_at: datetime


def _prepare_for_serving(model: Any) -> Any:
    """Un predict de una sola fila no gana nada con varios hilos del bosque."""
    final = getattr(model, "named_steps", {}).get("model")
    if final is not None and hasattr(final, "n_jobs"):
        final.set_params(n_jobs=1)
    return model


class ModelStore:
    """
    Guarda el modelo champion y lo mantiene al día.

    refresh() pregunta a MLflow qué versión tiene el alias y solo descarga si
    cambió. Es seguro llamarla desde varios hilos.
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        alias: str = config.MODEL_ALIAS,
        ttl_seconds: int = config.MODEL_CACHE_TTL_SECONDS,
        tracking_uri: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.alias = alias
        self.ttl_seconds = ttl_seconds
        self.tracking_uri = tracking_uri or config.MLFLOW_TRACKING_URI
        self._current: Optional[LoadedModel] = None
        self._lock = threading.Lock()
        self.last_error: Optional[str] = None
        self.last_check_at: Optional[datetime] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lectura -----------------------------------------------------------
    def snapshot(self) -> Optional[LoadedModel]:
        return self._current

    # -- actualización -----------------------------------------------------
    def refresh(self) -> bool:
        """Devuelve True si cargó una versión nueva."""
        with self._lock:
            try:
                client = MlflowClient(tracking_uri=self.tracking_uri)
                version = client.get_model_version_by_alias(self.model_name, self.alias)
                self.last_check_at = datetime.now(timezone.utc)

                current = self._current
                if current is not None and current.version == str(version.version):
                    self.last_error = None
                    return False

                mlflow.set_tracking_uri(self.tracking_uri)
                model = mlflow.sklearn.load_model(
                    f"models:/{self.model_name}/{version.version}"
                )
                loaded = LoadedModel(
                    model=_prepare_for_serving(model),
                    name=self.model_name,
                    version=str(version.version),
                    alias=self.alias,
                    run_id=version.run_id,
                    threshold=float((version.tags or {}).get("threshold", config.DECISION_THRESHOLD)),
                    loaded_at=datetime.now(timezone.utc),
                )
                self._current = loaded
                self.last_error = None
                self._publish_metrics(loaded)
                MODEL_RELOADS.labels("ok").inc()
                logger.info("Modelo cargado: %s v%s (alias %s, umbral %.3f)",
                            loaded.name, loaded.version, loaded.alias, loaded.threshold)
                return True
            except Exception as error:  # MLflow caído, alias inexistente, artefacto dañado...
                self.last_error = f"{type(error).__name__}: {error}"[:500]
                MODEL_RELOADS.labels("error").inc()
                if self._current is None:
                    logger.warning("Todavía no hay modelo disponible: %s", self.last_error)
                else:
                    logger.warning("No se pudo verificar el modelo; sigo con v%s: %s",
                                   self._current.version, self.last_error)
                return False

    @staticmethod
    def _publish_metrics(loaded: LoadedModel) -> None:
        MODEL_INFO.clear()  # solo una versión "activa" a la vez
        MODEL_INFO.labels(loaded.name, loaded.version, loaded.alias).set(1)
        MODEL_LOADED.set(1)
        MODEL_LOADED_AT.set(loaded.loaded_at.timestamp())
        try:
            MODEL_VERSION.set(int(loaded.version))
        except ValueError:
            MODEL_VERSION.set(0)

    # -- hilo de actualización ----------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="model-refresh", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while True:
            wait = self.ttl_seconds if self._current is not None \
                else min(self.ttl_seconds, RETRY_SECONDS_WITHOUT_MODEL)
            if self._stop.wait(wait):
                return
            self.refresh()


# ---------------------------------------------------------------------------
# Base de datos: registro de inferencias
# ---------------------------------------------------------------------------
INSERT_INFERENCE = text(
    f"""
    INSERT INTO {config.INFERENCE_TABLE}
        (request_id, created_at, input_data, status, prediction, probability, threshold,
         error_message, model_name, model_version, model_alias, latency_ms, source)
    VALUES
        (CAST(:request_id AS uuid), :created_at, CAST(:input_data AS jsonb), :status,
         :prediction, :probability, :threshold, :error_message, :model_name,
         :model_version, :model_alias, :latency_ms, :source)
    """
)


def build_engine() -> Optional[Engine]:
    """Engine de PostgreSQL, o None si la API no tiene base de datos configurada."""
    url = os.getenv("DATABASE_URL")
    if not url and config.POSTGRES_USER:
        url = config.postgres_url()
    if not url:
        logger.warning("Sin base de datos configurada: las inferencias NO se registrarán")
        return None
    return create_engine(url, pool_size=5, max_overflow=5, pool_pre_ping=True, future=True)


def write_inference(engine: Optional[Engine], record: Dict[str, Any]) -> None:
    """Guarda una inferencia; nunca lanza error (se corre después de responder)."""
    if engine is None:
        LOG_WRITES.labels("skipped").inc()
        return
    try:
        with engine.begin() as conn:
            conn.execute(INSERT_INFERENCE, record)
        LOG_WRITES.labels("ok").inc()
    except Exception:
        LOG_WRITES.labels("error").inc()
        logger.exception("No se pudo registrar la inferencia %s", record.get("request_id"))


# ---------------------------------------------------------------------------
# Middleware de métricas (ASGI puro: mide hasta el último byte enviado)
# ---------------------------------------------------------------------------
class MetricsMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status = {"code": 500}
        IN_PROGRESS.inc()

        async def send_wrapper(message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            IN_PROGRESS.dec()
            route = scope.get("route")
            endpoint = getattr(route, "path", None) or "unmatched"
            if endpoint != "/metrics":
                REQUESTS.labels(scope["method"], endpoint, str(status["code"])).inc()
                LATENCY.labels(endpoint).observe(time.perf_counter() - started)


# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------
def create_app(store: Optional[ModelStore] = None, engine: Optional[Engine] = None,
               use_default_engine: bool = True) -> FastAPI:
    """
    Arma la aplicación. Los parámetros existen para las pruebas; en producción
    se llama sin argumentos y todo viene de las variables de entorno.
    """
    state: Dict[str, Any] = {"store": store, "engine": engine}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                            format="%(asctime)s %(levelname)s %(name)s %(message)s")
        if state["store"] is None:
            state["store"] = ModelStore()
        if state["engine"] is None and use_default_engine:
            state["engine"] = build_engine()
        MODEL_LOADED.set(0)
        state["store"].refresh()      # primer intento antes de recibir tráfico
        state["store"].start()
        yield
        state["store"].stop()
        if state["engine"] is not None:
            state["engine"].dispose()

    api = FastAPI(
        title="API de inferencia: readmisión en pacientes con diabetes",
        description="Predice si un paciente será readmitido en menos de 30 días. "
                    "El modelo sale de MLflow (alias champion).",
        version="1.0.0",
        lifespan=lifespan,
    )
    api.add_middleware(MetricsMiddleware)

    @api.get("/health", response_model=schemas.HealthResponse, tags=["estado"])
    def health() -> schemas.HealthResponse:
        """La API está viva. No depende de MLflow ni de la base de datos."""
        return schemas.HealthResponse(model_loaded=state["store"].snapshot() is not None)

    @api.get("/ready", response_model=schemas.ReadyResponse, tags=["estado"])
    def ready() -> schemas.ReadyResponse:
        """Lista para recibir predicciones: hay un modelo cargado."""
        loaded = state["store"].snapshot()
        if loaded is None:
            raise HTTPException(status_code=503, detail="Todavía no hay un modelo cargado")
        return schemas.ReadyResponse(model_version=loaded.version)

    @api.get("/model-info", response_model=schemas.ModelInfoResponse, tags=["modelo"])
    def model_info() -> schemas.ModelInfoResponse:
        """Qué modelo está usando la API en este momento."""
        current_store: ModelStore = state["store"]
        loaded = current_store.snapshot()
        if loaded is None:
            status = "error" if current_store.last_error else "not_loaded"
        else:
            status = "loaded"
        return schemas.ModelInfoResponse(
            model_name=current_store.model_name,
            model_version=loaded.version if loaded else None,
            model_alias=current_store.alias,
            run_id=loaded.run_id if loaded else None,
            status=status,
            threshold=loaded.threshold if loaded else None,
            loaded_at=loaded.loaded_at if loaded else None,
            last_check_at=current_store.last_check_at,
            refresh_interval_seconds=current_store.ttl_seconds,
            last_error=current_store.last_error,
        )

    @api.post("/predict", response_model=schemas.PredictionResponse, tags=["inferencia"])
    def predict(
        patient: schemas.PatientFeatures,
        background: BackgroundTasks,
        x_client: Optional[str] = Header(default=None),
    ) -> schemas.PredictionResponse:
        """
        Predice la readmisión en menos de 30 días para un paciente.

        Un dato fuera del contrato (valor no permitido, campo faltante o de
        más) responde 422 con el detalle de cada error.
        """
        request_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        payload = patient.model_dump()
        source = (x_client or "unknown")[:50]
        current_store: ModelStore = state["store"]

        loaded = current_store.snapshot()
        if loaded is None:
            PREDICTION_ERRORS.labels("model_unavailable").inc()
            raise HTTPException(
                status_code=503,
                detail="No hay un modelo disponible todavía (¿ya corrió el DAG de entrenamiento?)",
            )

        try:
            inference_started = time.perf_counter()
            frame = pd.DataFrame([payload], columns=schemas.FEATURE_COLUMNS)
            probability = float(loaded.model.predict_proba(frame)[0, 1])
            INFERENCE_SECONDS.observe(time.perf_counter() - inference_started)
        except Exception as error:
            PREDICTION_ERRORS.labels("model_error").inc()
            logger.exception("Falló la predicción %s", request_id)
            background.add_task(write_inference, state["engine"], {
                "request_id": request_id, "created_at": created_at,
                "input_data": json.dumps(payload), "status": "error",
                "prediction": None, "probability": None, "threshold": loaded.threshold,
                "error_message": f"{type(error).__name__}: {error}"[:500],
                "model_name": loaded.name, "model_version": loaded.version,
                "model_alias": loaded.alias,
                "latency_ms": (time.perf_counter() - started) * 1000, "source": source,
            })
            raise HTTPException(status_code=500, detail="Error al ejecutar el modelo")

        prediction = int(probability >= loaded.threshold)
        elapsed_ms = (time.perf_counter() - started) * 1000
        PREDICTIONS.labels(str(prediction)).inc()
        PROBABILITY.observe(probability)

        background.add_task(write_inference, state["engine"], {
            "request_id": request_id, "created_at": created_at,
            "input_data": json.dumps(payload), "status": "success",
            "prediction": prediction, "probability": probability,
            "threshold": loaded.threshold, "error_message": None,
            "model_name": loaded.name, "model_version": loaded.version,
            "model_alias": loaded.alias, "latency_ms": elapsed_ms, "source": source,
        })

        return schemas.PredictionResponse(
            request_id=request_id,
            prediction=prediction,
            prediction_label=POSITIVE_LABEL if prediction else NEGATIVE_LABEL,
            probability=probability,
            threshold=loaded.threshold,
            model_name=loaded.name,
            model_version=loaded.version,
            model_alias=loaded.alias,
            processing_time_ms=round(elapsed_ms, 3),
        )

    @api.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        """Métricas para Prometheus."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return api


app = create_app()
