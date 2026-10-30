"""
Limpieza y partición de los datos (carril B).

Este módulo es lo que importa el DAG de Airflow (tarea "clean_data") y el
entrenamiento. No se conecta a ninguna base de datos: recibe un DataFrame y
devuelve otro. Quien lee de raw_data y escribe en clean_data es la tarea del
DAG, así este código se puede probar en cualquier máquina.

Reglas (justificadas en notebooks/01_eda.ipynb y docs/decisiones.md):
  1. "?" y vacío se tratan como nulo. El texto "None" NO es nulo: en A1Cresult
     y max_glu_serum significa "prueba no realizada".
  2. Se eliminan encuentros repetidos (mismo encounter_id).
  3. Se excluyen encuentros con gender inválido.
  4. Se excluyen encuentros de pacientes fallecidos o en hospicio: no pueden
     tener readmisión.
  5. race nulo -> "Unknown".
  6. Diagnósticos ICD-9 -> 9 grupos (Strack et al., 2014) + "Missing".
  7. medical_specialty (73 valores) -> 7 grupos.
  8. Target binario: 1 si readmitted == "<30", 0 en otro caso.
  9. Solo salen las columnas de CLEAN_COLUMNS (ids, features del contrato y
     target). Lo demás se descarta.

El split train/val/test es determinista por paciente (hash de patient_nbr),
así un paciente nunca queda en dos conjuntos aunque lleguen lotes nuevos.

Uso rápido para verificar con el CSV real:
    python3 src/preprocessing.py data/Diabetes/Diabetes.csv
"""

import hashlib
import logging
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import ValidationError

import config
import schemas

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
# discharge_disposition_id: 11, 19, 20, 21 = falleció; 13, 14 = hospicio.
EXPIRED_OR_HOSPICE_IDS = [11, 13, 14, 19, 20, 21]

VALID_GENDERS = ["Male", "Female"]
VALID_TARGET_VALUES = ["NO", ">30", "<30"]
NULL_MARKERS = ["?", ""]

ID_COLUMNS = ["encounter_id", config.PATIENT_ID_COLUMN]

# Columnas de clean_data, en este orden. Es el contrato con el carril A: la
# tabla clean_data debe tener exactamente estas columnas.
CLEAN_COLUMNS = ID_COLUMNS + schemas.FEATURE_COLUMNS + [config.TARGET_COLUMN]

# Columnas que se leen de raw_data.
REQUIRED_RAW_COLUMNS = (
    ID_COLUMNS + schemas.FEATURE_COLUMNS + [config.TARGET_SOURCE_COLUMN]
)

DIAG_COLUMNS = ["diag_1", "diag_2", "diag_3"]

# Especialidades que se conservan con su nombre; Surgery-* se junta en
# "Surgery" y el resto va a "Other".
KEPT_SPECIALTIES = [
    "InternalMedicine",
    "Emergency/Trauma",
    "Family/GeneralPractice",
    "Cardiology",
]

SPLIT_NAMES = ["train", "val", "test"]


# ---------------------------------------------------------------------------
# Agrupaciones
# ---------------------------------------------------------------------------
def _is_null(value) -> bool:
    """True para None, NaN o los marcadores de nulo del dataset."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip() in NULL_MARKERS or pd.isna(value)


def group_icd9(value) -> str:
    """
    Agrupa un código de diagnóstico ICD-9 en una de las categorías del
    contrato (schemas.DiagGroup), según Strack et al. (2014).

    Se compara la parte entera del código (459.81 -> 459) para que los
    códigos con decimales no se salgan del rango.
    """
    if _is_null(value):
        return "Missing"

    text = str(value).strip().upper()
    if text.startswith(("V", "E")):
        return "Other"
    try:
        code = int(float(text))
    except ValueError:
        return "Other"

    if 390 <= code <= 459 or code == 785:
        return "Circulatory"
    if 460 <= code <= 519 or code == 786:
        return "Respiratory"
    if 520 <= code <= 579 or code == 787:
        return "Digestive"
    if code == 250:
        return "Diabetes"
    if 800 <= code <= 999:
        return "Injury"
    if 710 <= code <= 739:
        return "Musculoskeletal"
    if 580 <= code <= 629 or code == 788:
        return "Genitourinary"
    if 140 <= code <= 239:
        return "Neoplasms"
    return "Other"


def group_specialty(value) -> str:
    """Agrupa medical_specialty (73 valores) en los 7 del contrato."""
    if _is_null(value):
        return "Unknown"

    text = str(value).strip()
    if text in KEPT_SPECIALTIES:
        return text
    if text.startswith("Surgery"):
        return "Surgery"
    return "Other"


# ---------------------------------------------------------------------------
# Limpieza
# ---------------------------------------------------------------------------
def _nulls_to_nan(series: pd.Series) -> pd.Series:
    """Convierte "?" y vacío en NaN sin tocar el texto "None"."""
    as_text = series.astype(str).str.strip()
    return series.mask(as_text.isin(NULL_MARKERS))


def clean_data(raw_df: pd.DataFrame, report: Optional[dict] = None) -> pd.DataFrame:
    """
    Convierte filas de raw_data en filas de clean_data.

    raw_df puede traer las columnas como texto o como números (por ejemplo,
    leídas de una tabla donde todo es TEXT). Si se pasa un diccionario en
    `report`, se llena con cuántas filas salieron en cada paso, para dejar
    evidencia en los logs del DAG.
    """
    rep = report if report is not None else {}

    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en los datos crudos: {missing}")

    df = raw_df[REQUIRED_RAW_COLUMNS].copy()
    rep["filas_entrada"] = len(df)

    # 1. Nulos
    for col in df.columns:
        df[col] = _nulls_to_nan(df[col])

    # El target debe traer solo los valores conocidos; si no, se avisa en
    # lugar de convertir en silencio cualquier cosa rara en "no readmitido".
    target_src = config.TARGET_SOURCE_COLUMN
    unexpected = set(df[target_src].dropna().unique()) - set(VALID_TARGET_VALUES)
    if unexpected or df[target_src].isna().any():
        raise ValueError(
            f"Valores inesperados en {target_src}: {sorted(map(str, unexpected))} "
            f"(nulos: {int(df[target_src].isna().sum())})"
        )

    # Los identificadores deben ser números.
    for col in ID_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")

    # 2. Encuentros repetidos
    before = len(df)
    df = df.drop_duplicates(subset="encounter_id", keep="first")
    rep["eliminadas_encuentro_repetido"] = before - len(df)

    # 3. Género inválido
    before = len(df)
    df = df[df["gender"].isin(VALID_GENDERS)].copy()
    rep["eliminadas_gender_invalido"] = before - len(df)

    # Columnas numéricas: texto -> número. Lo que no se pueda convertir queda
    # nulo y se elimina (en el dataset real no hay casos).
    for col in schemas.NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=schemas.NUMERIC_FEATURES).copy()
    rep["eliminadas_numericos_nulos"] = before - len(df)

    # 4. Fallecidos u hospicio
    before = len(df)
    df = df[~df["discharge_disposition_id"].isin(EXPIRED_OR_HOSPICE_IDS)].copy()
    rep["eliminadas_fallecido_hospicio"] = before - len(df)

    for col in schemas.NUMERIC_FEATURES:
        df[col] = df[col].astype("int64")

    # 5. race nulo
    df["race"] = df["race"].fillna("Unknown")

    # "None" es un valor real; si llegó como nulo (por ejemplo, porque
    # alguien leyó el CSV con los valores por defecto de pandas), se repone.
    for col in ["A1Cresult", "max_glu_serum"]:
        df[col] = df[col].fillna("None")

    # 6 y 7. Agrupaciones
    for col in DIAG_COLUMNS:
        df[col] = df[col].map(group_icd9)
    df["medical_specialty"] = df["medical_specialty"].map(group_specialty)

    # 8. Target
    df[config.TARGET_COLUMN] = (
        df[target_src] == config.POSITIVE_CLASS_VALUE
    ).astype("int16")

    result = df[CLEAN_COLUMNS].reset_index(drop=True)
    rep["filas_salida"] = len(result)
    rep["positivos"] = int(result[config.TARGET_COLUMN].sum())

    logger.info("clean_data: %s", rep)
    return result


def validate_against_contract(
    clean_df: pd.DataFrame, max_samples: int = 5
) -> Tuple[int, List[str]]:
    """
    Comprueba que cada fila limpia cumpla schemas.PatientFeatures, es decir,
    que la API aceptaría ese paciente. Devuelve (filas inválidas, ejemplos
    de errores).
    """
    invalid = 0
    samples: List[str] = []
    records = clean_df[schemas.FEATURE_COLUMNS].to_dict("records")
    for position, row in enumerate(records):
        try:
            schemas.PatientFeatures(**row)
        except ValidationError as error:
            invalid += 1
            if len(samples) < max_samples:
                fields = ", ".join(str(e["loc"][0]) for e in error.errors())
                samples.append(f"fila {position}: {fields}")
    return invalid, samples


# ---------------------------------------------------------------------------
# Split por paciente
# ---------------------------------------------------------------------------
def split_of(patient_id) -> str:
    """
    Asigna un paciente a "train", "val" o "test" con un hash de su id.

    El resultado depende solo del id y de config.SPLIT_SALT: no hay azar, así
    un paciente siempre cae en el mismo conjunto en todas las ejecuciones.
    """
    key = str(int(float(patient_id)))
    digest = hashlib.md5(f"{config.SPLIT_SALT}-{key}".encode()).hexdigest()
    position = int(digest[:8], 16) / 0xFFFFFFFF  # número entre 0 y 1

    if position < config.TRAIN_FRACTION:
        return "train"
    if position < config.TRAIN_FRACTION + config.VAL_FRACTION:
        return "val"
    return "test"


def split_dataset(clean_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Separa clean_data en train, val y test sin partir ningún paciente."""
    labels = clean_df[config.PATIENT_ID_COLUMN].map(split_of)
    return {
        name: clean_df[labels == name].reset_index(drop=True)
        for name in SPLIT_NAMES
    }


# ---------------------------------------------------------------------------
# Verificación desde la terminal
# ---------------------------------------------------------------------------
def _main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("Uso: python3 src/preprocessing.py ruta/al/Diabetes.csv")
        return 2

    logging.basicConfig(level=logging.WARNING)

    # Todo como texto, igual que podría venir de una tabla cruda.
    raw = pd.read_csv(argv[1], dtype=str, keep_default_na=False)
    report: dict = {}
    clean = clean_data(raw, report)

    print("\n=== Limpieza ===")
    for name, value in report.items():
        print(f"{name:34s} {value:>9,}")

    invalid, samples = validate_against_contract(clean)
    print("\n=== Contrato (schemas.PatientFeatures) ===")
    print(f"filas que la API rechazaría       {invalid:>9,}")
    for sample in samples:
        print("  ", sample)

    parts = split_dataset(clean)
    print("\n=== Split por paciente ===")
    print(f"{'conjunto':10s}{'filas':>10s}{'pacientes':>11s}{'% positivos':>13s}")
    for name, part in parts.items():
        rate = part[config.TARGET_COLUMN].mean() * 100
        patients = part[config.PATIENT_ID_COLUMN].nunique()
        print(f"{name:10s}{len(part):>10,}{patients:>11,}{rate:>12.2f}%")

    shared = (
        set(parts["train"][config.PATIENT_ID_COLUMN])
        & set(parts["val"][config.PATIENT_ID_COLUMN])
        | set(parts["train"][config.PATIENT_ID_COLUMN])
        & set(parts["test"][config.PATIENT_ID_COLUMN])
        | set(parts["val"][config.PATIENT_ID_COLUMN])
        & set(parts["test"][config.PATIENT_ID_COLUMN])
    )
    print(f"pacientes en más de un conjunto   {len(shared):>9,}")

    return 1 if invalid or shared else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
