# Locust con el contrato de la API (para generar solicitudes válidas).
#   docker build -f locust/Dockerfile -t dnino97/locust-diabetes:latest .
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN pip install locust==2.32.5 pydantic==2.13.5
COPY src/schemas.py ./
COPY locust/locustfile.py ./
RUN useradd --create-home appuser
USER appuser
EXPOSE 8089
# La interfaz web queda en :8089. El host se escribe en la web o con --host.
CMD ["locust", "-f", "/app/locustfile.py"]
