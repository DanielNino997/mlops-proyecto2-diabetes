# MLOps Proyecto 2: readmisión de pacientes con diabetes

Proyecto 2 (Nivel 3) del curso **Operaciones de Machine Learning**, Pontificia Universidad Javeriana.

Sistema MLOps completo desplegado en Kubernetes: los datos clínicos llegan por lotes, se guardan crudos, se limpian, entrenan un modelo que se registra en MLflow, y ese modelo se sirve por una API que se consulta desde una interfaz web. Todo con pruebas de carga y observabilidad.

**Problema:** predecir si un paciente con diabetes será readmitido en el hospital en menos de 30 días (clasificación binaria).

**Dataset:** Diabetes 130-US hospitals (1999-2008), versión modificada entregada por el curso.

> **Estado:** en construcción. Entrega: 2 de noviembre de 2026.

---

## Equipo

| Carril | Responsable | Componentes |
|---|---|---|
| A · Plataforma | Ana María Merchán León | Kubernetes, PostgreSQL, MinIO, MLflow, Airflow, Prometheus, Grafana, imágenes en DockerHub |
| B · ML y aplicación | Daniel Niño | EDA, preprocesamiento, entrenamiento, API FastAPI, Streamlit, Locust |
| Conjunto | Ambos | Contratos entre componentes, DAG de Airflow, pruebas de carga, documentación y video |

---

## Arquitectura

```
                    FASE DE ENTRENAMIENTO
 Diabetes.csv ──> Airflow (DAG) ──> PostgreSQL: raw_data ──> clean_data
                       │
                       └──> entrena ──> MLflow ──> metadatos en PostgreSQL
                                          │        artefactos en MinIO
                                          │
                    FASE DE INFERENCIA    │ (modelo con alias "champion")
                                          v
 Streamlit ─────────────────────────> API FastAPI ──> PostgreSQL: inference_logs
 Locust (carga) ────────────────────>     │
                                          └── /metrics ──> Prometheus ──> Grafana
```

Todos los componentes corren como contenedores en un clúster local de Kubernetes.

---

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| `src/` | Código compartido: contratos (`config.py`, `schemas.py`) y lógica de datos y entrenamiento |
| `sql/` | Creación de tablas (`init.sql`) |
| `notebooks/` | Análisis exploratorio (EDA) que justifica las decisiones de limpieza |
| `airflow/dags/` | DAG que orquesta carga, limpieza, entrenamiento y promoción del modelo |
| `api/` | API de inferencia (FastAPI) |
| `ui/` | Interfaz gráfica (Streamlit) |
| `locust/` | Escenario de pruebas de carga |
| `k8s/` | Manifiestos de Kubernetes |
| `monitoring/` | Configuración de Prometheus y dashboard de Grafana |
| `docs/` | Decisiones (`decisiones.md`), métricas (`metricas.md`), pruebas de carga, evidencias y guion de presentación |

---

## Contratos entre componentes

Para que las piezas encajen, la forma en que se comunican está escrita una sola vez en el código y todos la importan:

| Archivo | Qué define |
|---|---|
| `src/config.py` | Nombre del modelo en MLflow, alias `champion`, nombres de tablas y Services. Todo se lee de variables de entorno (ConfigMap y Secret en Kubernetes). |
| `src/schemas.py` | Datos que recibe y devuelve `/predict`, respuestas de `/model-info` y `/health`, lista de features del modelo y un paciente de ejemplo. |
| `sql/init.sql` | Las tres capas de datos: `raw_data`, `clean_data` e `inference_logs`. |

Cualquier cambio a un contrato se revisa entre los dos integrantes.

---

## Cómo ejecutar el EDA

Requiere Python 3.9 o superior. Desde la raíz del repo:

```
python3 -m venv .venv
```

```
source .venv/bin/activate
```

```
pip install pydantic pandas matplotlib nbconvert ipykernel
```

```
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

El notebook descarga el dataset en `data/Diabetes/` (esa carpeta no se sube al repo).

---

## Carril B: ejecutar y probar en local

Pruebas automáticas (103). Las que usan PostgreSQL necesitan `TEST_DATABASE_URL` (**borran** las tablas de esa base: usar una de pruebas):

```
export TEST_DATABASE_URL=postgresql+psycopg2://usuario:clave@localhost:5432/diabetes_data
```

```
python3 -m pytest -q
```

Entrenar sin Airflow con el CSV (registra en MLflow y promueve a `champion`):

```
python3 src/train.py --csv data/Diabetes/Diabetes.csv --tracking-uri sqlite:///mlflow.db
```

API (necesita un MLflow con un modelo `champion`) y Streamlit:

```
MLFLOW_TRACKING_URI=sqlite:///mlflow.db uvicorn --app-dir api inference_api:app --port 8000
```

```
API_URL=http://localhost:8000 streamlit run ui/streamlit_app.py
```

Pruebas de carga: ver `docs/pruebas_de_carga.md`. Métricas para Grafana: `docs/metricas.md`.

### Imágenes (contexto de build = raíz del repo)

| Imagen | Comando |
|---|---|
| Airflow + DAG | `docker build -f airflow/Dockerfile -t dnino97/airflow-diabetes:latest .` |
| API | `docker build -f api/Dockerfile -t dnino97/inference-api:latest .` |
| Streamlit | `docker build -f ui/Dockerfile -t dnino97/streamlit-ui:latest .` |
| Locust | `docker build -f locust/Dockerfile -t dnino97/locust-diabetes:latest .` |

### Variables de entorno que esperan los componentes

| Variable | Dónde | Para qué |
|---|---|---|
| `POSTGRES_HOST/PORT/DB/USER/PASSWORD` | Airflow, API | Conexión a PostgreSQL (usuario y clave desde el Secret) |
| `MLFLOW_TRACKING_URI` | Airflow, API | Servidor MLflow (ej. `http://mlflow:5000`) |
| `MLFLOW_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Airflow, API | Acceso a los artefactos en MinIO |
| `SOURCE_FILE` | Airflow | Ruta del CSV (por defecto `/opt/airflow/data/Diabetes/Diabetes.csv`) |
| `BATCH_SIZE` | Airflow | Filas por lote (15000) |
| `DAG_SCHEDULE` | Airflow | Frecuencia del DAG (`@daily`) |
| `MODEL_CACHE_TTL_SECONDS` | API | Cada cuánto revisa si hay un champion nuevo (60) |
| `API_URL` | Streamlit | Dirección de la API (`http://inference-api:8000`) |

El DAG crea las tablas solo (`init_schema` ejecuta `sql/init.sql`). La API expone `/health` (liveness) y `/ready` (readiness: solo responde `ready` con el modelo cargado).

---

## Despliegue en Kubernetes

*Pendiente: se documentará cuando el clúster esté integrado.*

---

## Checklist de aceptación (sección 9 del enunciado)

- [ ] Todo desplegado en Kubernetes (no solo docker-compose)
- [ ] El DAG corre y muestra varias tareas independientes
- [ ] Datos crudos y procesados en tablas separadas
- [ ] Carga por lotes de máximo 15.000 registros
- [ ] MLflow con backend en PostgreSQL y artefactos en MinIO (sin SQLite)
- [ ] La API usa el modelo que viene de MLflow
- [ ] La API no tiene una ruta local quemada al modelo
- [ ] Cada inferencia queda registrada en la base de datos
- [ ] La API expone `/metrics`
- [ ] Prometheus recolecta las métricas de la API
- [ ] Grafana muestra el efecto de una prueba con Locust
- [ ] Cada contenedor tiene `requests` y `limits`
- [ ] Las imágenes propias están en DockerHub
