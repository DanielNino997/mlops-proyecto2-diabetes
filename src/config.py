"""
Configuración compartida del Proyecto 2 (contrato entre carril A y carril B).

Todo lo que cambia entre "mi máquina" y "el clúster" se lee de variables de
entorno. En Kubernetes esas variables vienen del ConfigMap (lo no sensible) y
del Secret (credenciales). En local se pueden poner en un archivo .env.

Regla: ningún otro archivo escribe a mano nombres de tablas, del modelo o de
Services. Todos importan este módulo.
"""

import os
from typing import Optional


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Lee una variable de entorno; si no existe, usa el valor por defecto."""
    return os.getenv(name, default)


# ---------------------------------------------------------------------------
# Modelo en MLflow
# ---------------------------------------------------------------------------
# Nombre con el que el DAG registra el modelo y con el que la API lo busca.
MODEL_NAME = _env("MODEL_NAME", "diabetes-readmission")

# Alias del modelo productivo. La API siempre carga el que tenga este alias.
MODEL_ALIAS = _env("MODEL_ALIAS", "champion")

# Experimento donde el DAG registra cada entrenamiento.
EXPERIMENT_NAME = _env("EXPERIMENT_NAME", "diabetes-readmission")

# Cada cuántos segundos la API pregunta a MLflow qué versión tiene el alias.
# Solo descarga el modelo si la versión cambió (caché con TTL).
MODEL_CACHE_TTL_SECONDS = int(_env("MODEL_CACHE_TTL_SECONDS", "60"))


# ---------------------------------------------------------------------------
# Conexión a MLflow y MinIO
# ---------------------------------------------------------------------------
# Con artefactos en MinIO no basta MLFLOW_TRACKING_URI: el cliente de MLflow
# también sube y descarga los archivos del modelo directamente a MinIO, así
# que necesita su dirección y credenciales (y boto3 instalado).
MLFLOW_TRACKING_URI = _env("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_S3_ENDPOINT_URL = _env("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY_ID = _env("AWS_ACCESS_KEY_ID")          # viene del Secret
AWS_SECRET_ACCESS_KEY = _env("AWS_SECRET_ACCESS_KEY")  # viene del Secret
MLFLOW_ARTIFACT_BUCKET = _env("MLFLOW_ARTIFACT_BUCKET", "mlflow-artifacts")


# ---------------------------------------------------------------------------
# PostgreSQL de datos (raw, clean e inferencias)
# ---------------------------------------------------------------------------
# Es la base de datos del proyecto, distinta de la que usa MLflow para sus
# metadatos.
POSTGRES_HOST = _env("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(_env("POSTGRES_PORT", "5432"))
POSTGRES_DB = _env("POSTGRES_DB", "mlops_data")
POSTGRES_USER = _env("POSTGRES_USER")          # viene del Secret
POSTGRES_PASSWORD = _env("POSTGRES_PASSWORD")  # viene del Secret


def postgres_url() -> str:
    """URL de conexión para SQLAlchemy (se arma al usarla, no al importar)."""
    return (
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )


# Nombres de las tres capas de datos (deben coincidir con sql/init.sql).
RAW_TABLE = _env("RAW_TABLE", "raw_data")
CLEAN_TABLE = _env("CLEAN_TABLE", "clean_data")
INFERENCE_TABLE = _env("INFERENCE_TABLE", "inference_logs")


# ---------------------------------------------------------------------------
# Datos fuente y carga por lotes
# ---------------------------------------------------------------------------
SOURCE_FILE = _env("SOURCE_FILE", "/opt/airflow/data/Diabetes/Diabetes.csv")

# El enunciado exige lotes de máximo 15.000 registros.
BATCH_SIZE = int(_env("BATCH_SIZE", "15000"))


# ---------------------------------------------------------------------------
# Columnas del archivo fuente (50), en el orden del CSV. raw_data las guarda
# todas como TEXT, tal como llegan, sin transformarlas.
# ---------------------------------------------------------------------------
SOURCE_COLUMNS = [
    "encounter_id", "patient_nbr", "race", "gender", "age", "weight",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "payer_code", "medical_specialty",
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide",
    "chlorpropamide", "glimepiride", "acetohexamide", "glipizide",
    "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone", "change", "diabetesMed", "readmitted",
]

# Tabla de control de lotes (qué filas del archivo corresponden a cada lote).
BATCH_TABLE = _env("BATCH_TABLE", "ingestion_batches")

# Estados de una fila de raw_data a lo largo del DAG.
RAW_STATUS_LOADED = "loaded"        # recién insertada
RAW_STATUS_VALIDATED = "validated"  # pasó la validación de calidad
RAW_STATUS_REJECTED = "rejected"    # falló la validación (se conserva para auditoría)
RAW_STATUS_PROCESSED = "processed"  # ya está en clean_data
RAW_STATUS_EXCLUDED = "excluded"    # válida, pero excluida en la limpieza


# ---------------------------------------------------------------------------
# Problema de ML
# ---------------------------------------------------------------------------
# PENDIENTE EDA: confirmar nombre de la columna y sus valores en el dataset
# modificado del curso.
TARGET_SOURCE_COLUMN = "readmitted"   # valores originales: "NO", ">30", "<30"
POSITIVE_CLASS_VALUE = "<30"          # 1 = readmitido en menos de 30 días
TARGET_COLUMN = "readmitted_lt30"     # columna binaria (0/1) en clean_data

# Columna que se usa para separar train/val/test por paciente.
PATIENT_ID_COLUMN = "patient_nbr"

# Métrica principal para elegir y promover el modelo. Opciones: "f2", "recall",
# "f1", "pr_auc", "roc_auc". Por defecto F2: pesa el recall el doble que la
# precisión (perder a un paciente que sí se readmite cuesta más que una alerta
# de más) y, a diferencia de recall solo, no premia a un modelo que marque a
# todos como readmitidos.
PRIMARY_METRIC = _env("PRIMARY_METRIC", "f2")

# El umbral de decisión de cada modelo se elige en validación para maximizar
# F-beta con este beta (2 = el recall pesa el doble). Se guarda como tag de la
# versión del modelo en MLflow y la API lo usa al predecir.
THRESHOLD_BETA = float(_env("THRESHOLD_BETA", "2.0"))

# Umbral de respaldo si el modelo no trae uno guardado.
DECISION_THRESHOLD = float(_env("DECISION_THRESHOLD", "0.5"))

# Un candidato solo reemplaza al champion si mejora la métrica principal, en el
# mismo conjunto de prueba, al menos en esta cantidad. 0 = basta con mejorar.
MIN_IMPROVEMENT = float(_env("MIN_IMPROVEMENT", "0.0"))

# Mínimo de positivos que debe tener cada conjunto para entrenar y evaluar.
MIN_POSITIVES_PER_SPLIT = int(_env("MIN_POSITIVES_PER_SPLIT", "30"))

RANDOM_STATE = int(_env("RANDOM_STATE", "42"))

# Núcleos que puede usar el entrenamiento (el pod de Airflow tiene límite de CPU).
TRAIN_N_JOBS = int(_env("TRAIN_N_JOBS", "2"))

# Proporciones del split por paciente (deben sumar 1).
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# Semilla del hash que asigna cada paciente a train/val/test. Si se cambia,
# los pacientes cambian de conjunto: no tocar después de congelar contratos.
SPLIT_SALT = "proyecto2-mlops-2026"


# ---------------------------------------------------------------------------
# Services de Kubernetes que usa la interfaz
# ---------------------------------------------------------------------------
# Streamlit solo habla con la API, nunca con MLflow ni con Postgres.
API_URL = _env("API_URL", "http://inference-api:8000")
