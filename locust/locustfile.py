"""
Pruebas de carga sobre la API de inferencia (Locust).

Uso local:
    locust -f locust/locustfile.py --host http://localhost:8000
    # sin interfaz web, 50 usuarios, 5 nuevos por segundo, 2 minutos:
    locust -f locust/locustfile.py --host http://localhost:8000 --headless \
           -u 50 -r 5 -t 2m --csv resultados/r1_50u

Escenario por etapas (para hallar el punto de degradación):
    LOAD_SHAPE=steps locust -f locust/locustfile.py --host http://... --headless --csv resultados/pasos

Variables de entorno:
    INVALID_RATIO   fracción de solicitudes inválidas a propósito (default 0.0)
    WAIT_MIN/WAIT_MAX  pausa entre solicitudes de un usuario, en segundos (default 1 / 2.5, igual que el taller de Locust; no cambiarlo entre 1 y 3 réplicas)
    STEP_USERS / STEP_SECONDS / STEP_MAX_USERS   forma por etapas (default 20 / 60 / 200)
"""

import os
import random
import sys
from pathlib import Path

from locust import HttpUser, LoadTestShape, between, task

# schemas.py está en ../src (en la imagen de Docker se copia junto a este archivo)
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

import schemas  # noqa: E402

INVALID_RATIO = float(os.getenv("INVALID_RATIO", "0"))
WAIT_MIN = float(os.getenv("WAIT_MIN", "1"))
WAIT_MAX = float(os.getenv("WAIT_MAX", "2.5"))
HEADERS = {"X-Client": "locust"}

# Rangos tomados del contrato: si el contrato cambia, la prueba se ajusta sola.
FIELDS = schemas.PatientFeatures.model_fields


def _options(name):
    from typing import get_args
    return list(get_args(FIELDS[name].annotation))


def _range(name):
    lo, hi = 0, 1
    for meta in FIELDS[name].metadata:
        lo = getattr(meta, "ge", lo)
        hi = getattr(meta, "le", hi)
    return lo, hi


# Valores "típicos" para que los números no sean extremos en cada solicitud.
_TYPICAL_MAX = {"num_lab_procedures": 90, "num_medications": 40, "number_outpatient": 5,
                "number_emergency": 4, "number_inpatient": 6, "num_procedures": 6,
                "discharge_disposition_id": 25}


def random_patient():
    """Un paciente válido y distinto en cada llamada."""
    row = {}
    for name in schemas.FEATURE_COLUMNS:
        options = _options(name)
        if options:
            row[name] = random.choice(options)
        else:
            lo, hi = _range(name)
            row[name] = random.randint(lo, min(hi, _TYPICAL_MAX.get(name, hi)))
    return row


def invalid_patient():
    """Un paciente inválido a propósito (la API debe responder 422)."""
    row = random_patient()
    kind = random.choice(["category", "missing", "range", "extra"])
    if kind == "category":
        row["age"] = "70"
    elif kind == "missing":
        row.pop("race")
    elif kind == "range":
        row["time_in_hospital"] = 99
    else:
        row["campo_inventado"] = 1
    return row


class PredictionUser(HttpUser):
    wait_time = between(WAIT_MIN, WAIT_MAX)

    @task(20)
    def predict(self):
        if INVALID_RATIO and random.random() < INVALID_RATIO:
            self.invalid()
            return
        with self.client.post("/predict", json=random_patient(), headers=HEADERS,
                              name="/predict", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
                return
            try:
                body = response.json()
            except ValueError:
                response.failure("Respuesta sin JSON válido")
                return
            if "prediction" not in body or "probability" not in body:
                response.failure("Falta 'prediction' o 'probability' en la respuesta")
                return
            response.success()

    @task(1)
    def model_info(self):
        self.client.get("/model-info", name="/model-info")

    def invalid(self):
        # 422 es el comportamiento correcto: no se cuenta como fallo.
        with self.client.post("/predict", json=invalid_patient(), headers=HEADERS,
                              name="/predict (inválida)", catch_response=True) as response:
            if response.status_code == 422:
                response.success()
            else:
                response.failure(f"se esperaba 422 y llegó {response.status_code}")


if os.getenv("LOAD_SHAPE") == "steps":

    class StepLoad(LoadTestShape):
        """Sube STEP_USERS usuarios cada STEP_SECONDS hasta STEP_MAX_USERS."""
        step_users = int(os.getenv("STEP_USERS", "20"))
        step_seconds = int(os.getenv("STEP_SECONDS", "60"))
        max_users = int(os.getenv("STEP_MAX_USERS", "200"))

        def tick(self):
            step = int(self.get_run_time() // self.step_seconds) + 1
            users = step * self.step_users
            if users > self.max_users:
                return None
            return users, self.step_users
