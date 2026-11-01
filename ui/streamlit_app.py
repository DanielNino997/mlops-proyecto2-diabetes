"""
Interfaz gráfica (Streamlit) para consumir la API de inferencia.

Permite: ingresar valores, cargar valores de ejemplo, enviar la solicitud a la
API, ver la predicción, ver la versión del modelo usada y ver los errores de
validación cuando los datos no son válidos.

Ejecutar en local (la API debe estar corriendo):
    API_URL=http://localhost:8000 streamlit run ui/streamlit_app.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, get_args

import pandas as pd
import streamlit as st

# En el repositorio, schemas.py vive en ../src; en la imagen de Docker se copia
# junto a este archivo.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

import api_client  # noqa: E402
import schemas  # noqa: E402

API_URL = os.getenv("API_URL", "http://inference-api:8000")

st.set_page_config(page_title="Riesgo de readmisión", page_icon="🩺", layout="wide")

# ---------------------------------------------------------------------------
# Textos en español para cada campo del contrato
# ---------------------------------------------------------------------------
LABELS: Dict[str, str] = {
    "race": "Raza", "gender": "Sexo", "age": "Edad (rango)",
    "admission_type_id": "Tipo de ingreso (código)",
    "discharge_disposition_id": "Destino al alta (código)",
    "admission_source_id": "Origen del ingreso (código)",
    "time_in_hospital": "Días en el hospital",
    "medical_specialty": "Especialidad del médico",
    "num_lab_procedures": "Pruebas de laboratorio", "num_procedures": "Procedimientos",
    "num_medications": "Medicamentos administrados", "number_diagnoses": "Diagnósticos registrados",
    "number_outpatient": "Visitas ambulatorias (año previo)",
    "number_emergency": "Visitas a urgencias (año previo)",
    "number_inpatient": "Hospitalizaciones (año previo)",
    "diag_1": "Diagnóstico principal", "diag_2": "Diagnóstico secundario",
    "diag_3": "Diagnóstico adicional",
    "max_glu_serum": "Glucosa en suero", "A1Cresult": "Hemoglobina glicosilada (A1C)",
    "insulin": "Insulina", "metformin": "Metformina",
    "change": "¿Cambió la medicación?", "diabetesMed": "¿Recibe medicamentos para diabetes?",
}

SECTIONS = [
    ("Paciente", ["race", "gender", "age"]),
    ("Ingreso y estancia", ["admission_type_id", "discharge_disposition_id",
                            "admission_source_id", "time_in_hospital", "medical_specialty"]),
    ("Atención durante la estancia", ["num_lab_procedures", "num_procedures",
                                      "num_medications", "number_diagnoses"]),
    ("Visitas en el año previo", ["number_outpatient", "number_emergency", "number_inpatient"]),
    ("Diagnósticos", ["diag_1", "diag_2", "diag_3"]),
    ("Laboratorio y medicación", ["max_glu_serum", "A1Cresult", "insulin", "metformin",
                                  "change", "diabetesMed"]),
]

LOW_RISK = schemas.EXAMPLE_PATIENT.model_dump()
HIGH_RISK = {**LOW_RISK, "number_inpatient": 5, "number_emergency": 3, "number_outpatient": 2,
             "time_in_hospital": 9, "num_medications": 32, "number_diagnoses": 9,
             "insulin": "Up", "A1Cresult": ">8", "diag_1": "Diabetes", "age": "[80-90)"}
EXAMPLES = {"Ejemplo 1: riesgo habitual": LOW_RISK, "Ejemplo 2: riesgo alto": HIGH_RISK}

FIELDS = schemas.PatientFeatures.model_fields


def bounds(name: str) -> Dict[str, int]:
    """Mínimo y máximo permitidos por el contrato para un campo numérico."""
    lo = hi = None
    for meta in FIELDS[name].metadata:
        lo = getattr(meta, "ge", lo)
        hi = getattr(meta, "le", hi)
    return {"min_value": lo, "max_value": hi}


def load_example(values: Dict[str, Any]) -> None:
    for key, value in values.items():
        st.session_state[f"f_{key}"] = value


if "f_race" not in st.session_state:
    load_example(LOW_RISK)

# ---------------------------------------------------------------------------
# Barra lateral: estado de la API y del modelo
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Estado del servicio")
    info = api_client.get_model_info(API_URL)
    if info is None:
        st.error("La API no responde")
    elif info["status"] != "loaded":
        st.warning("API activa, pero sin modelo cargado")
        if info.get("last_error"):
            st.caption(info["last_error"])
    else:
        st.success("API y modelo en línea")
        st.metric("Versión del modelo", f"v{info['model_version']}")
        st.caption(f"Modelo: {info['model_name']}  \nAlias: {info['model_alias']}")
        st.caption(f"Umbral de decisión: {info['threshold']:.3f}")
    st.caption(f"API: {API_URL}")
    st.button("Actualizar estado")

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("🩺 Riesgo de readmisión en menos de 30 días")
st.write(
    "Pacientes con diabetes: ingrese los datos del encuentro hospitalario y la API "
    "estimará si el paciente será readmitido dentro de los 30 días siguientes al alta."
)


def show_result(result: api_client.PredictResult, sent: Dict[str, Any]) -> None:
    """Muestra la predicción o, si falló, qué pasó."""
    if result.ok and result.data:
        data = result.data
        positive = data["prediction"] == 1
        box = st.error if positive else st.success
        box(f"**{data['prediction_label']}**")
        left, middle, right = st.columns(3)
        left.metric("Probabilidad de readmisión", f"{data['probability'] * 100:.1f} %")
        middle.metric("Umbral de decisión", f"{data['threshold'] * 100:.1f} %")
        right.metric("Versión del modelo", f"v{data['model_version']}")
        st.progress(min(max(data["probability"], 0.0), 1.0))
        st.caption(
            f"Modelo `{data['model_name']}` (alias `{data['model_alias']}`) · "
            f"tiempo de procesamiento {data['processing_time_ms']:.1f} ms · "
            f"solicitud `{data['request_id']}`"
        )
        with st.expander("Ver la respuesta completa de la API"):
            st.json(data)
        return

    if result.status_code == 422:
        st.error(f"**{result.message}** La API rechazó la solicitud; corrija lo siguiente:")
        st.dataframe(pd.DataFrame(result.errors), hide_index=True, use_container_width=True)
    else:
        st.error(result.message or "No se pudo obtener la predicción.")
    with st.expander("Ver lo que se envió a la API"):
        st.json(sent)


tab_form, tab_json = st.tabs(["Formulario", "JSON manual (avanzado)"])

# ---------------------------------------------------------------------------
# Pestaña 1: formulario
# ---------------------------------------------------------------------------
with tab_form:
    st.caption("Cargar valores de ejemplo:")
    columns = st.columns(len(EXAMPLES) + 2)
    for column, (title, values) in zip(columns, EXAMPLES.items()):
        column.button(title, on_click=load_example, args=(values,), use_container_width=True)

    with st.form("paciente"):
        payload: Dict[str, Any] = {}
        for title, names in SECTIONS:
            st.subheader(title)
            cols = st.columns(3)
            for index, name in enumerate(names):
                column = cols[index % 3]
                options = get_args(FIELDS[name].annotation)
                if options:
                    payload[name] = column.selectbox(
                        LABELS[name], options=list(options), key=f"f_{name}")
                else:
                    payload[name] = int(column.number_input(
                        LABELS[name], step=1, key=f"f_{name}", **bounds(name)))
        submitted = st.form_submit_button("Calcular riesgo", type="primary")

    if submitted:
        with st.spinner("Consultando la API..."):
            outcome = api_client.post_predict(API_URL, payload)
        show_result(outcome, payload)

# ---------------------------------------------------------------------------
# Pestaña 2: JSON manual, para probar datos inválidos
# ---------------------------------------------------------------------------
with tab_json:
    st.write(
        "Envíe el JSON tal cual a la API. Sirve para ver cómo responde ante datos "
        "inválidos: cambie, por ejemplo, `\"age\"` por `\"70\"` o borre un campo."
    )
    raw_text = st.text_area("JSON de la solicitud", value=json.dumps(LOW_RISK, indent=2),
                            height=420, key="raw_json")
    if st.button("Enviar JSON"):
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as error:
            st.error(f"El texto no es un JSON válido: {error}")
        else:
            with st.spinner("Consultando la API..."):
                outcome = api_client.post_predict(API_URL, parsed)
            show_result(outcome, parsed if isinstance(parsed, dict) else {})
