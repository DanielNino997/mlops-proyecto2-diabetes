# Imagen de la interfaz Streamlit (solo habla con la API por HTTP).
#   docker build -f ui/Dockerfile -t dnino97/streamlit-ui:latest .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY ui/requirements.txt .
RUN pip install -r requirements.txt

COPY src/schemas.py ./
COPY ui/streamlit_app.py ui/api_client.py ./

RUN useradd --create-home appuser
USER appuser

ENV API_URL=http://inference-api:8000
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
