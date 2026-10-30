"""
Contrato de la API de inferencia (entrada y salida de /predict y /model-info).

Este archivo lo importan:
  - la API (FastAPI valida cada solicitud con PatientFeatures),
  - el entrenamiento (FEATURE_COLUMNS define qué columnas entran al modelo),
  - Streamlit (EXAMPLE_PATIENT es el botón "cargar valores de ejemplo").

Decisión de diseño: la API recibe valores "humanos" ya limpios (categorías
agrupadas), no columnas codificadas. El one-hot y el escalado van dentro del
Pipeline de sklearn que se registra en MLflow, así la API nunca repite
transformaciones.

PENDIENTE EDA: confirmar la lista de features y los valores permitidos de
cada categoría contra el dataset modificado del curso.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Valores permitidos de las variables categóricas
# ---------------------------------------------------------------------------
AgeBracket = Literal[
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]
Race = Literal["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"]
Gender = Literal["Male", "Female"]

# Diagnósticos ICD-9 agrupados en categorías (la agrupación vive en
# src/preprocessing.py y se aplica antes de guardar en clean_data).
DiagGroup = Literal[
    "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
    "Musculoskeletal", "Genitourinary", "Neoplasms", "Other",
]

# Especialidad del médico, agrupada (las poco frecuentes van a "Other").
SpecialtyGroup = Literal[
    "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice",
    "Cardiology", "Surgery", "Other", "Unknown",
]

LabResult = Literal["None", "Norm", ">7", ">8"]          # A1Cresult
GlucoseResult = Literal["None", "Norm", ">200", ">300"]  # max_glu_serum
MedChange = Literal["No", "Steady", "Up", "Down"]        # insulin, metformin
YesNo = Literal["Yes", "No"]


# ---------------------------------------------------------------------------
# Entrada de /predict
# ---------------------------------------------------------------------------
class PatientFeatures(BaseModel):
    """Datos de un encuentro hospitalario que recibe /predict."""

    # Rechaza campos que no estén en el contrato, para que un error de
    # nombre no pase en silencio.
    model_config = ConfigDict(extra="forbid")

    # Demografía
    race: Race
    gender: Gender
    age: AgeBracket

    # Ingreso y alta (códigos numéricos del dataset)
    admission_type_id: int = Field(ge=1, le=8)
    discharge_disposition_id: int = Field(ge=1, le=29)
    admission_source_id: int = Field(ge=1, le=26)

    # Estancia y atención
    time_in_hospital: int = Field(ge=1, le=14)
    medical_specialty: SpecialtyGroup
    num_lab_procedures: int = Field(ge=0, le=150)
    num_procedures: int = Field(ge=0, le=10)
    num_medications: int = Field(ge=0, le=100)
    number_diagnoses: int = Field(ge=1, le=20)

    # Visitas en el año anterior
    number_outpatient: int = Field(ge=0, le=50)
    number_emergency: int = Field(ge=0, le=80)
    number_inpatient: int = Field(ge=0, le=25)

    # Diagnósticos agrupados
    diag_1: DiagGroup
    diag_2: DiagGroup
    diag_3: DiagGroup

    # Laboratorio y medicamentos
    max_glu_serum: GlucoseResult
    A1Cresult: LabResult
    insulin: MedChange
    metformin: MedChange
    change: Literal["Ch", "No"]
    diabetesMed: YesNo


# Lista de columnas en el orden en que entran al modelo. El entrenamiento la
# usa para seleccionar columnas de clean_data; así API y modelo nunca difieren.
FEATURE_COLUMNS: list[str] = list(PatientFeatures.model_fields.keys())

NUMERIC_FEATURES: list[str] = [
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_diagnoses", "number_outpatient",
    "number_emergency", "number_inpatient",
]
# PENDIENTE EDA: los tres *_id son códigos, no cantidades. Decidir si se
# tratan como categóricos (one-hot) y moverlos a CATEGORICAL_FEATURES.

CATEGORICAL_FEATURES: list[str] = [
    c for c in FEATURE_COLUMNS if c not in NUMERIC_FEATURES
]


# Valores de ejemplo para el botón de Streamlit y para el archivo de Locust.
EXAMPLE_PATIENT = PatientFeatures(
    race="Caucasian",
    gender="Female",
    age="[70-80)",
    admission_type_id=1,
    discharge_disposition_id=1,
    admission_source_id=7,
    time_in_hospital=5,
    medical_specialty="InternalMedicine",
    num_lab_procedures=45,
    num_procedures=1,
    num_medications=18,
    number_diagnoses=9,
    number_outpatient=0,
    number_emergency=0,
    number_inpatient=1,
    diag_1="Circulatory",
    diag_2="Diabetes",
    diag_3="Circulatory",
    max_glu_serum="None",
    A1Cresult=">8",
    insulin="Up",
    metformin="No",
    change="Ch",
    diabetesMed="Yes",
)


# ---------------------------------------------------------------------------
# Salida de /predict
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    """Respuesta de /predict (campos mínimos exigidos por el enunciado)."""

    model_config = ConfigDict(protected_namespaces=())

    request_id: str = Field(description="UUID de la solicitud; mismo valor que en inference_logs")
    prediction: int = Field(ge=0, le=1, description="1 = readmisión en menos de 30 días")
    prediction_label: str
    probability: float = Field(ge=0, le=1, description="Probabilidad de la clase positiva")
    threshold: float = Field(ge=0, le=1, description="Umbral usado para convertir la probabilidad en 0/1")
    model_name: str
    model_version: str
    model_alias: str
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Salida de /model-info
# ---------------------------------------------------------------------------
class ModelInfoResponse(BaseModel):
    """Respuesta de /model-info: qué modelo está usando la API ahora."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: Optional[str]
    model_alias: str
    run_id: Optional[str]
    status: Literal["loaded", "not_loaded", "error"]
    loaded_at: Optional[datetime]
    features: list[str] = FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Salida de /health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_loaded: bool
