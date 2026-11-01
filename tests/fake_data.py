"""
Generador de datos sintéticos con el mismo formato del dataset del curso.

Sirve para probar el pipeline completo (carga, limpieza, entrenamiento, API)
sin descargar el CSV real. Los datos NO son clínicos: son inventados, con una
señal débil pero real (más hospitalizaciones previas -> más riesgo) para que
los modelos tengan algo que aprender.

Se parece al original en lo que importa para las pruebas:
  * 50 columnas con los mismos nombres y orden;
  * "?" como marca de nulo, y el texto "None" como valor real;
  * pacientes con varios encuentros;
  * unos pocos fallecidos/hospicio y género inválido;
  * ~11% de readmitidos en menos de 30 días.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config

AGES = [f"[{i}-{i + 10})" for i in range(0, 100, 10)]
AGE_WEIGHTS = np.array([0.2, 0.7, 1.6, 3.7, 9.5, 17, 22, 26, 17, 2.7])
AGE_WEIGHTS = AGE_WEIGHTS / AGE_WEIGHTS.sum()

MEDS = [c for c in config.SOURCE_COLUMNS if c in (
    "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
    "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone",
    "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone")]

DIAG_POOL = ["250.83", "250.01", "428", "414", "401", "786", "V57", "E888",
             "584", "715", "153", "996", "276", "486", "574", "585", "459.81",
             "519.1", "8", "?"]
DIAG_P = np.array([4, 6, 12, 10, 9, 6, 2, 1, 4, 3, 3, 2, 6, 6, 3, 3, 1, 1, 2, 1], float)
DIAG_P = DIAG_P / DIAG_P.sum()

SPECIALTIES = ["?", "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice",
               "Cardiology", "Surgery-General", "Nephrology", "Orthopedics",
               "Surgery-Cardiovascular/Thoracic", "Psychiatry"]
SPEC_P = np.array([49, 14, 7, 7, 5, 3, 2, 2, 1, 1], float)
SPEC_P = SPEC_P / SPEC_P.sum()


def make_fake_dataframe(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Devuelve n encuentros sintéticos; todas las columnas como texto."""
    rng = np.random.default_rng(seed)
    pick = lambda values, p=None: rng.choice(values, n, p=p)

    n_patients = max(int(n * 0.7), 1)
    inpatient = rng.poisson(0.6, n)
    emergency = rng.poisson(0.25, n)
    outpatient = rng.poisson(0.4, n)
    days = rng.integers(1, 15, n)
    meds = np.clip(rng.normal(16, 8, n).round(), 1, 80).astype(int)
    insulin = pick(["No", "Steady", "Down", "Up"], [0.47, 0.30, 0.12, 0.11])
    diag_1 = pick(DIAG_POOL, DIAG_P)
    age = pick(AGES, AGE_WEIGHTS)

    logit = (
        -3.9
        + 0.65 * np.minimum(inpatient, 6)
        + 0.30 * np.minimum(emergency, 5)
        + 0.06 * days
        + 0.025 * meds
        + 0.45 * (insulin == "Up")
        + 0.40 * np.char.startswith(diag_1.astype(str), "250")
        + 0.30 * np.isin(age, AGES[7:])
    )
    p_early = 1 / (1 + np.exp(-logit))
    draw = rng.random(n)
    readmitted = np.where(
        draw < p_early, "<30",
        np.where(rng.random(n) < 0.4, ">30", "NO"),
    )

    discharge = pick([1, 3, 6, 18, 22, 25, 11, 13, 14],
                     [0.58, 0.14, 0.12, 0.05, 0.05, 0.03, 0.015, 0.008, 0.007])
    gender = pick(["Female", "Male", "Unknown/Invalid"], [0.537, 0.4625, 0.0005])
    data = {
        "encounter_id": 2278392 + np.arange(n) * 7,
        "patient_nbr": 1000 + rng.integers(0, n_patients, n) * 13,
        "race": pick(["Caucasian", "AfricanAmerican", "?", "Hispanic", "Other", "Asian"],
                     [0.74, 0.19, 0.022, 0.02, 0.015, 0.013]),
        "gender": gender,
        "age": age,
        "weight": pick(["?", "[75-100)", "[50-75)"], [0.968, 0.02, 0.012]),
        "admission_type_id": pick([1, 2, 3, 5, 6, 8], [0.53, 0.18, 0.18, 0.05, 0.05, 0.01]),
        "discharge_disposition_id": discharge,
        "admission_source_id": pick([1, 7, 17, 4, 6, 2], [0.29, 0.56, 0.07, 0.03, 0.03, 0.02]),
        "time_in_hospital": days,
        "payer_code": pick(["?", "MC", "HM", "SP", "BC"], [0.4, 0.32, 0.06, 0.05, 0.17]),
        "medical_specialty": pick(SPECIALTIES, SPEC_P),
        "num_lab_procedures": np.clip(rng.normal(43, 19, n).round(), 1, 132).astype(int),
        "num_procedures": rng.choice([0, 1, 2, 3, 4, 5, 6], n, p=[.46, .20, .12, .09, .05, .05, .03]),
        "num_medications": meds,
        "number_outpatient": outpatient,
        "number_emergency": emergency,
        "number_inpatient": inpatient,
        "diag_1": diag_1,
        "diag_2": pick(DIAG_POOL, DIAG_P),
        "diag_3": pick(DIAG_POOL, DIAG_P),
        "number_diagnoses": rng.integers(1, 17, n),
        "max_glu_serum": pick(["None", "Norm", ">200", ">300"], [0.947, 0.025, 0.015, 0.013]),
        "A1Cresult": pick(["None", ">8", "Norm", ">7"], [0.833, 0.081, 0.049, 0.037]),
        "metformin": pick(["No", "Steady", "Up", "Down"], [0.80, 0.18, 0.01, 0.01]),
        "insulin": insulin,
        "change": pick(["Ch", "No"], [0.46, 0.54]),
        "diabetesMed": pick(["Yes", "No"], [0.77, 0.23]),
        "readmitted": readmitted,
    }
    for med in MEDS:
        data[med] = np.full(n, "No", dtype=object)

    frame = pd.DataFrame(data)[config.SOURCE_COLUMNS]
    return frame.astype(str)


def write_fake_csv(path: str, n: int = 5000, seed: int = 0) -> Path:
    """Escribe el CSV sintético y devuelve su ruta."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    make_fake_dataframe(n, seed).to_csv(target, index=False)
    return target


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "fake_diabetes.csv"
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    write_fake_csv(out, rows)
    print(f"Escribí {rows} filas en {out}")
