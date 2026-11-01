# Imagen de la API de inferencia.
# Contexto de build = raíz del repositorio:
#   docker build -f api/Dockerfile -t dnino97/inference-api:latest .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY api/requirements.txt .
RUN pip install -r requirements.txt

COPY src/config.py src/schemas.py ./
COPY api/inference_api.py .

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
# Un solo proceso por pod: se escala con réplicas de Kubernetes, no con workers.
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
