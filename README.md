# MLOps Proyecto 2: readmisión de pacientes con diabetes

Proyecto 2 (Nivel 3) del curso **Operaciones de Machine Learning**, Pontificia Universidad Javeriana.

Sistema MLOps completo desplegado en Kubernetes: los datos clínicos llegan por lotes, se guardan crudos, se limpian, entrenan un modelo que se registra en MLflow, y ese modelo se sirve por una API que se consulta desde una interfaz web. Todo con pruebas de carga y observabilidad.

**Problema:** predecir si un paciente con diabetes será readmitido en el hospital en menos de 30 días (clasificación binaria).

**Dataset:** Diabetes 130-US hospitals (1999-2008), versión modificada entregada por el curso.

> **Estado:** completo y desplegado en Kubernetes. Entrega: 2 de noviembre de 2026.

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
| Airflow + DAG | `docker build -f airflow/Dockerfile -t annamerchanlm/mlops-p2-airflow:3.0.6-v1 .` |
| MLflow | `docker build -t annamerchanlm/mlops-p2-mlflow:2.16.2 mlflow/` |
| API | `docker build -f api/Dockerfile -t dnino97/diabetes-api:1.0 .` |
| Streamlit | `docker build -f ui/Dockerfile -t dnino97/diabetes-ui:1.0 .` |
| Locust | `docker build -f locust/Dockerfile -t dnino97/diabetes-locust:1.0 .` |

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

Clúster: k3s embebido en Rancher (contenedor `rancher/rancher` en la VM, sin permisos de sudo).
Dos namespaces: `mlops` (plataforma y aplicación) y `airflow-local` (Airflow, como en la guía del curso).

### 0. Prerrequisitos del clúster

- `kubectl` y `helm` en `~/.local/bin`, con el kubeconfig del clúster `local` de Rancher en `~/.kube/config`.
- StorageClass por defecto `local-path` (el k3s de Rancher no la trae):

```bash
git clone --depth 1 --branch v0.0.31 git@github.com:rancher/local-path-provisioner.git
sed -i 's#/opt/local-path-provisioner#/var/lib/rancher/local-path#' local-path-provisioner/deploy/local-path-storage.yaml
docker exec -i rancher kubectl apply -f - < local-path-provisioner/deploy/local-path-storage.yaml
kubectl patch storageclass local-path -p '{"metadata":{"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```

### 1. Datos y modelos (namespace `mlops`)

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/postgres/            # PostgreSQL (bases mlops_data y mlflow) + Adminer
kubectl exec -i -n mlops deploy/postgres -- psql -U mlops -d mlops_data < sql/init.sql
kubectl apply -f k8s/minio/
kubectl exec -n mlops deploy/minio -- sh -c 'mc alias set local http://localhost:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD && mc mb --ignore-existing local/mlflow-artifacts'
kubectl apply -f k8s/mlflow/              # MLflow 2.16.2: backend PostgreSQL, artefactos en MinIO
```

### 2. Airflow (Helm, namespace `airflow-local`)

```bash
helm repo add apache-airflow https://airflow.apache.org && helm repo update
kubectl create namespace airflow-local
kubectl apply -f k8s/airflow/config.yaml  # variables del DAG (hosts con FQDN + Secret propio)
helm install airflow apache-airflow/airflow -n airflow-local -f k8s/airflow/values.yaml --wait --timeout 15m
```

La imagen `annamerchanlm/mlops-p2-airflow:3.0.6-v1` (construida con `airflow/Dockerfile`) trae el DAG, `src/`, `sql/` y el CSV.

### 3. Inferencia y observabilidad (namespace `mlops`)

```bash
kubectl apply -f k8s/api/api.yaml         # 2 réplicas, carga el modelo con alias champion
kubectl apply -f k8s/ui/ui.yaml           # Streamlit
kubectl apply -f k8s/monitoring/prometheus.yaml   # descubre cada pod de la API (RBAC + kubernetes_sd)
kubectl create configmap grafana-dashboards -n mlops \
  --from-file=api-dashboard.json=monitoring/grafana/api-dashboard.json --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/monitoring/grafana.yaml
kubectl apply -f k8s/locust/locust.yaml
```

### 4. Acceso a las interfaces

El firewall de la VM solo permite SSH: se usa `kubectl port-forward` en la VM y el reenvío de puertos de VS Code (Remote-SSH) hacia el equipo local.

| Servicio | Comando | URL local | Credenciales |
|---|---|---|---|
| Airflow | `kubectl port-forward svc/airflow-api-server 8082:8080 -n airflow-local` | http://localhost:8082 | admin / admin |
| MLflow | `kubectl port-forward svc/mlflow 5000:5000 -n mlops` | http://localhost:5000 | — |
| API | `kubectl port-forward svc/inference-api 8000:8000 -n mlops` | http://localhost:8000/docs | — |
| Streamlit | `kubectl port-forward svc/streamlit 8501:8501 -n mlops` | http://localhost:8501 | — |
| Grafana | `kubectl port-forward svc/grafana 3000:3000 -n mlops` | http://localhost:3000 | admin / grafana_pass_2026 |
| Prometheus | `kubectl port-forward svc/prometheus 9090:9090 -n mlops` | http://localhost:9090 | — |
| Locust | `kubectl port-forward svc/locust 8089:8089 -n mlops` | http://localhost:8089 | — |
| Adminer | `kubectl port-forward svc/adminer 8081:8080 -n mlops` | http://localhost:8081 | mlops / mlops_pass_2026 (servidor `postgres`) |

### 5. Validación rápida

```bash
kubectl get pods -n mlops && kubectl get pods -n airflow-local
kubectl exec -n airflow-local deploy/airflow-dag-processor -c dag-processor -- airflow dags trigger diabetes_pipeline
kubectl exec -n mlops deploy/postgres -- psql -U mlops -d mlops_data -c \
  "SELECT 'raw', count(*) FROM raw_data UNION ALL SELECT 'clean', count(*) FROM clean_data UNION ALL SELECT 'batches', count(*) FROM ingestion_batches;"
curl -s localhost:8000/model-info    # con el port-forward de la API activo
```

Resultado de referencia (7 ejecuciones del DAG): `raw_data` 101.766 filas, `clean_data` 99.340, 7 lotes; modelo `diabetes-readmission` v7 (random_forest) con alias `champion`.

### Imágenes publicadas en DockerHub

| Imagen | Componente |
|---|---|
| `annamerchanlm/mlops-p2-mlflow:2.16.2` | Servidor MLflow (`mlflow/Dockerfile`) |
| `annamerchanlm/mlops-p2-airflow:3.0.6-v1` | Airflow con DAG, código y datos (`airflow/Dockerfile`) |
| `dnino97/diabetes-api:1.0` | API de inferencia (`api/Dockerfile`) |
| `dnino97/diabetes-ui:1.0` | Streamlit (`ui/Dockerfile`) |
| `dnino97/diabetes-locust:1.0` | Locust (`locust/Dockerfile`) |

---

## Checklist de aceptación (sección 9 del enunciado)

| # | Condición | Estado | Evidencia |
|---|---|---|---|
| 1 | Todo desplegado en Kubernetes | ✅ | `kubectl get pods -n mlops` y `-n airflow-local` |
| 2 | El DAG corre y muestra varias tareas independientes | ✅ | 7 ejecuciones, 11 tareas cada una; 3 entrenamientos en paralelo |
| 3 | Datos crudos y procesados en tablas separadas | ✅ | `raw_data` (101.766) y `clean_data` (99.340) |
| 4 | Carga por lotes de máximo 15.000 registros | ✅ | `ingestion_batches`: 7 lotes (6 × 15.000 + 11.766) |
| 5 | MLflow con backend PostgreSQL y artefactos en MinIO | ✅ | Base `mlflow` en PostgreSQL; bucket `mlflow-artifacts` (244 objetos) |
| 6 | La API usa el modelo que viene de MLflow | ✅ | `/model-info`: `diabetes-readmission` v7, alias `champion` |
| 7 | La API no tiene una ruta local quemada | ✅ | Carga por alias; pasó de v2 a v7 sola, 8 s después de la promoción |
| 8 | Cada inferencia queda registrada | ✅ | `inference_logs`: 38.697 filas de Locust registradas bajo carga |
| 9 | La API expone `/metrics` | ✅ | Métricas `diabetes_api_*` (`docs/metricas.md`) |
| 10 | Prometheus recolecta las métricas | ✅ | Targets `up` por cada pod de `inference-api` |
| 11 | Grafana muestra el efecto de Locust | ✅ | Dashboard "API de inferencia - Diabetes" (`monitoring/grafana/api-dashboard.json`) |
| 12 | Cada contenedor tiene `requests` y `limits` | ✅ | 18 contenedores, incluidos los sidecars de Airflow |
| 13 | Imágenes propias en DockerHub | ✅ | Ver tabla de imágenes en "Despliegue en Kubernetes" |