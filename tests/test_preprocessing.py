"""
Pruebas de src/preprocessing.py.

Se corren desde la raíz del repo con:
    python3 -m pytest -q
"""

from typing import get_args

import numpy as np
import pandas as pd
import pytest

import config
import preprocessing as pp
import schemas


# ---------------------------------------------------------------------------
# Datos de apoyo
# ---------------------------------------------------------------------------
def raw_row(**overrides) -> dict:
    """Una fila cruda válida; cada prueba cambia solo lo que le interesa."""
    row = {
        "encounter_id": 1, "patient_nbr": 100,
        "race": "Caucasian", "gender": "Female", "age": "[70-80)",
        "weight": "?", "payer_code": "?",              # columnas que se descartan
        "repaglinide": "No", "glipizide": "Steady",    # medicamentos que se descartan
        "admission_type_id": 1, "discharge_disposition_id": 1,
        "admission_source_id": 7, "time_in_hospital": 5,
        "medical_specialty": "InternalMedicine",
        "num_lab_procedures": 45, "num_procedures": 1, "num_medications": 18,
        "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 1,
        "diag_1": "428", "diag_2": "250.01", "diag_3": "401",
        "number_diagnoses": 9, "max_glu_serum": "None", "A1Cresult": ">8",
        "metformin": "No", "insulin": "Up", "change": "Ch", "diabetesMed": "Yes",
        "readmitted": "<30",
    }
    row.update(overrides)
    return row


def raw_frame(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Agrupación de diagnósticos
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("code, expected", [
    ("250.83", "Diabetes"), ("250", "Diabetes"),
    ("428", "Circulatory"), ("785", "Circulatory"),
    ("459.81", "Circulatory"),            # decimales en el límite superior
    ("486", "Respiratory"), ("786", "Respiratory"), ("519.1", "Respiratory"),
    ("574", "Digestive"), ("787", "Digestive"), ("579.3", "Digestive"),
    ("585", "Genitourinary"), ("788", "Genitourinary"), ("629.1", "Genitourinary"),
    ("715", "Musculoskeletal"), ("739.1", "Musculoskeletal"),
    ("153", "Neoplasms"), ("239.9", "Neoplasms"),
    ("820", "Injury"), ("999.1", "Injury"),
    ("276", "Other"), ("8", "Other"), ("V57", "Other"), ("E888", "Other"),
    ("  428 ", "Circulatory"),            # espacios
    (None, "Missing"), (np.nan, "Missing"), ("?", "Missing"), ("", "Missing"),
])
def test_group_icd9(code, expected):
    assert pp.group_icd9(code) == expected


def test_group_icd9_only_returns_contract_values():
    allowed = set(get_args(schemas.DiagGroup))
    for code in ["250.01", "428", "V27", "E812", "8", "999", "xyz", None, "?"]:
        assert pp.group_icd9(code) in allowed


# ---------------------------------------------------------------------------
# Agrupación de especialidades
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value, expected", [
    ("InternalMedicine", "InternalMedicine"),
    ("Emergency/Trauma", "Emergency/Trauma"),
    ("Family/GeneralPractice", "Family/GeneralPractice"),
    ("Cardiology", "Cardiology"),
    ("Surgery-General", "Surgery"),
    ("Surgery-Cardiovascular/Thoracic", "Surgery"),
    ("Nephrology", "Other"),
    ("Cardiology-Pediatric", "Other"),
    (np.nan, "Unknown"), ("?", "Unknown"), (None, "Unknown"),
])
def test_group_specialty(value, expected):
    assert pp.group_specialty(value) == expected
    assert pp.group_specialty(value) in get_args(schemas.SpecialtyGroup)


# ---------------------------------------------------------------------------
# Limpieza
# ---------------------------------------------------------------------------
def test_clean_data_columns_and_contract():
    clean = pp.clean_data(raw_frame([raw_row()]))
    assert list(clean.columns) == pp.CLEAN_COLUMNS
    assert clean.loc[0, config.TARGET_COLUMN] == 1
    invalid, _ = pp.validate_against_contract(clean)
    assert invalid == 0


def test_clean_data_filters_and_reports():
    rows = [
        raw_row(encounter_id=1, patient_nbr=100),                                  # válida
        raw_row(encounter_id=2, patient_nbr=101, gender="Unknown/Invalid"),       # género inválido
        raw_row(encounter_id=3, patient_nbr=102, discharge_disposition_id=11),    # falleció
        raw_row(encounter_id=4, patient_nbr=103, discharge_disposition_id=14),    # hospicio
        raw_row(encounter_id=5, patient_nbr=104, race="?"),                       # race nulo
        raw_row(encounter_id=1, patient_nbr=100),                                  # encuentro repetido
    ]
    report: dict = {}
    clean = pp.clean_data(raw_frame(rows), report)

    assert sorted(clean["encounter_id"]) == [1, 5]
    assert clean.set_index("encounter_id").loc[5, "race"] == "Unknown"
    assert report["filas_entrada"] == 6
    assert report["eliminadas_encuentro_repetido"] == 1
    assert report["eliminadas_gender_invalido"] == 1
    assert report["eliminadas_fallecido_hospicio"] == 2
    assert report["filas_salida"] == 2


def test_clean_data_groups_and_none_values():
    rows = [
        raw_row(encounter_id=1, medical_specialty="Surgery-General"),
        raw_row(encounter_id=2, medical_specialty="Nephrology"),
        raw_row(encounter_id=3, medical_specialty="?"),
        raw_row(encounter_id=4, diag_2="?", diag_3=np.nan),
        raw_row(encounter_id=5, A1Cresult=np.nan, max_glu_serum=np.nan),
        raw_row(encounter_id=6, A1Cresult="None"),
    ]
    clean = pp.clean_data(raw_frame(rows)).set_index("encounter_id")

    assert clean.loc[1, "medical_specialty"] == "Surgery"
    assert clean.loc[2, "medical_specialty"] == "Other"
    assert clean.loc[3, "medical_specialty"] == "Unknown"
    assert clean.loc[4, "diag_2"] == "Missing" and clean.loc[4, "diag_3"] == "Missing"
    # "None" es un valor real (prueba no realizada), no un nulo.
    assert clean.loc[5, "A1Cresult"] == "None" and clean.loc[5, "max_glu_serum"] == "None"
    assert clean.loc[6, "A1Cresult"] == "None"
    assert pp.validate_against_contract(clean.reset_index())[0] == 0


def test_clean_data_target_mapping():
    rows = [
        raw_row(encounter_id=1, readmitted="<30"),
        raw_row(encounter_id=2, readmitted=">30"),
        raw_row(encounter_id=3, readmitted="NO"),
    ]
    clean = pp.clean_data(raw_frame(rows)).set_index("encounter_id")
    assert clean[config.TARGET_COLUMN].tolist() == [1, 0, 0]


def test_clean_data_accepts_everything_as_text():
    """Una tabla cruda donde todo es TEXT debe dar el mismo resultado."""
    rows = [raw_row(encounter_id=i, patient_nbr=100 + i) for i in range(1, 6)]
    as_numbers = pp.clean_data(raw_frame(rows))
    as_text = pp.clean_data(raw_frame(rows).astype(str))
    pd.testing.assert_frame_equal(as_numbers, as_text)


def test_clean_data_fails_loudly():
    with pytest.raises(ValueError, match="Faltan columnas"):
        pp.clean_data(raw_frame([raw_row()]).drop(columns=["age"]))
    with pytest.raises(ValueError, match="Valores inesperados"):
        pp.clean_data(raw_frame([raw_row(readmitted="maybe")]))


def test_validate_against_contract_detects_bad_values():
    clean = pp.clean_data(raw_frame([raw_row(encounter_id=1), raw_row(encounter_id=2)]))
    clean.loc[1, "age"] = "70"                 # no es un rango válido
    invalid, samples = pp.validate_against_contract(clean)
    assert invalid == 1 and "age" in samples[0]


# ---------------------------------------------------------------------------
# Split por paciente
# ---------------------------------------------------------------------------
def test_split_of_is_stable():
    """
    Valores de referencia. Si esta prueba falla, cambió SPLIT_SALT, las
    fracciones o la fórmula, y los pacientes cambiarían de conjunto: no
    mezclar con modelos ya entrenados.
    """
    expected = {
        8222157: "train", 55629189: "train", 86047875: "train",
        82442376: "val", 42519267: "train",
    }
    for patient_id, split in expected.items():
        assert pp.split_of(patient_id) == split


def test_split_of_accepts_int_str_and_float():
    assert pp.split_of(82442376) == pp.split_of("82442376") == pp.split_of(82442376.0)


def test_split_proportions():
    labels = pd.Series(range(1, 30001)).map(pp.split_of).value_counts(normalize=True)
    assert abs(labels["train"] - config.TRAIN_FRACTION) < 0.01
    assert abs(labels["val"] - config.VAL_FRACTION) < 0.01
    assert abs(labels["test"] - config.TEST_FRACTION) < 0.01


def test_split_dataset_never_splits_a_patient():
    rng = np.random.default_rng(0)
    rows = [
        raw_row(encounter_id=i, patient_nbr=int(rng.integers(1, 80)))
        for i in range(1, 400)
    ]
    clean = pp.clean_data(raw_frame(rows))
    parts = pp.split_dataset(clean)

    assert sum(len(p) for p in parts.values()) == len(clean)
    sets = {name: set(p[config.PATIENT_ID_COLUMN]) for name, p in parts.items()}
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])
