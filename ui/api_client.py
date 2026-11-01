"""
Cliente de la API para la interfaz. No usa Streamlit, así se puede probar solo.

La interfaz habla ÚNICAMENTE con la API (requests HTTP). No importa mlflow ni
se conecta a la base de datos: así lo exige el enunciado.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

CLIENT_NAME = "streamlit"


@dataclass
class PredictResult:
    ok: bool
    status_code: Optional[int] = None
    data: Optional[Dict[str, Any]] = None
    errors: List[Dict[str, str]] = field(default_factory=list)  # errores de validación (422)
    message: str = ""                                             # otros fallos, en lenguaje claro


def friendly_errors(detail: Any, sent: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """
    Convierte el detalle de un 422 de FastAPI en filas legibles:
    campo, qué está mal y qué valor se envió.
    """
    sent = sent or {}
    if not isinstance(detail, list):
        return [{"campo": "-", "problema": str(detail), "valor enviado": ""}]

    rows = []
    for error in detail:
        location = [str(part) for part in error.get("loc", []) if part != "body"]
        field_name = location[-1] if location else "-"
        kind = error.get("type", "")
        ctx = error.get("ctx", {}) or {}

        if kind == "missing":
            problem = "Falta este campo"
        elif kind == "literal_error":
            problem = f"Valor no permitido. Opciones: {ctx.get('expected', '?')}"
        elif kind in ("greater_than_equal", "greater_than"):
            problem = f"Debe ser mayor o igual a {ctx.get('ge', ctx.get('gt', '?'))}"
        elif kind in ("less_than_equal", "less_than"):
            problem = f"Debe ser menor o igual a {ctx.get('le', ctx.get('lt', '?'))}"
        elif kind.startswith("int_") or kind == "int_parsing":
            problem = "Debe ser un número entero"
        elif kind == "extra_forbidden":
            problem = "Campo no reconocido por la API"
        elif kind in ("json_invalid", "model_attributes_type"):
            problem = "El JSON enviado no es válido"
        else:
            problem = error.get("msg", "Valor inválido")

        value = error.get("input", sent.get(field_name, ""))
        rows.append({"campo": field_name, "problema": problem, "valor enviado": str(value)})
    return rows


def post_predict(base_url: str, payload: Any, timeout: float = 15.0) -> PredictResult:
    """Envía un paciente a /predict y traduce cualquier resultado a algo mostrable."""
    url = base_url.rstrip("/") + "/predict"
    try:
        response = requests.post(url, json=payload, timeout=timeout,
                                 headers={"X-Client": CLIENT_NAME})
    except requests.exceptions.Timeout:
        return PredictResult(False, message=f"La API no respondió en {timeout:.0f} segundos.")
    except requests.exceptions.RequestException as error:
        return PredictResult(False, message=f"No se pudo conectar con la API en {base_url}. ({error.__class__.__name__})")

    try:
        body = response.json()
    except ValueError:
        body = None

    if response.status_code == 200 and isinstance(body, dict):
        return PredictResult(True, 200, data=body)
    if response.status_code == 422 and isinstance(body, dict):
        sent = payload if isinstance(payload, dict) else {}
        return PredictResult(False, 422, errors=friendly_errors(body.get("detail"), sent),
                             message="Los datos ingresados no son válidos.")
    if response.status_code == 503:
        detail = body.get("detail") if isinstance(body, dict) else None
        return PredictResult(False, 503, message=detail or "El modelo todavía no está disponible.")
    detail = body.get("detail") if isinstance(body, dict) else response.text[:200]
    return PredictResult(False, response.status_code, message=f"Error {response.status_code}: {detail}")


def get_model_info(base_url: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    """Datos del modelo en uso, o None si la API no responde."""
    try:
        response = requests.get(base_url.rstrip("/") + "/model-info", timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except (requests.exceptions.RequestException, ValueError):
        pass
    return None
