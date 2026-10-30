# mlops-proyecto2-diabetes
MLOps Proyecto 2: readmisión de pacientes con diabetes
Proyecto 2 (Nivel 3) del curso Operaciones de Machine Learning, Pontificia Universidad Javeriana.
Sistema MLOps completo desplegado en Kubernetes: los datos clínicos llegan por lotes, se guardan crudos, se limpian, entrenan un modelo que se registra en MLflow, y ese modelo se sirve por una API que se consulta desde una interfaz web. Todo con pruebas de carga y observabilidad.
Problema: predecir si un paciente con diabetes será readmitido en el hospital en menos de 30 días (clasificación binaria).
Dataset: Diabetes 130-US hospitals (1999-2008), versión modificada entregada por el curso.
> **Estado:** en construcción. Entrega: 2 de noviembre de 2026.
---
Equipo
Carril	Responsable	Componentes
A · Plataforma	Ana María Merchán León	Kubernetes, PostgreSQL, MinIO, MLflow, Airflow, Prometheus, Grafana, imágenes en DockerHub
B · ML y aplicación	Daniel Niño	EDA, preprocesamiento, entrenamiento, API FastAPI, Streamlit, Locust
Conjunto	Ambos	Contratos entre componentes, DAG de Airflow, pruebas de carga, documentación y video
---
Arquitectura
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
Estructura del repositorio
Carpeta	Contenido
`src/`	Código compartido: contratos (`config.py`, `schemas.py`) y lógica de datos y entrenamiento
`sql/`	Creación de tablas (`init.sql`)
`notebooks/`	Análisis exploratorio (EDA) que justifica las decisiones de limpieza
`airflow/dags/`	DAG que orquesta carga, limpieza, entrenamiento y promoción del modelo
`api/`	API de inferencia (FastAPI)
`ui/`	Interfaz gráfica (Streamlit)
`locust/`	Escenario de pruebas de carga
`k8s/`	Manifiestos de Kubernetes
`monitoring/`	Configuración de Prometheus y dashboard de Grafana
`docs/`	Registro de decisiones del proyecto (`decisiones.md`)
---
Contratos entre componentes
Para que las piezas encajen, la forma en que se comunican está escrita una sola vez en el código y todos la importan:
Archivo	Qué define
`src/config.py`	Nombre del modelo en MLflow, alias `champion`, nombres de tablas y Services. Todo se lee de variables de entorno (ConfigMap y Secret en Kubernetes).
`src/schemas.py`	Datos que recibe y devuelve `/predict`, respuestas de `/model-info` y `/health`, lista de features del modelo y un paciente de ejemplo.
`sql/init.sql`	Las tres capas de datos: `raw_data`, `clean_data` e `inference_logs`.
Cualquier cambio a un contrato se revisa entre los dos integrantes.
---
Cómo ejecutar el EDA
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
Despliegue en Kubernetes
Pendiente: se documentará cuando el clúster esté integrado.
---
Checklist de aceptación (sección 9 del enunciado)
[ ] Todo desplegado en Kubernetes (no solo docker-compose)
[ ] El DAG corre y muestra varias tareas independientes
[ ] Datos crudos y procesados en tablas separadas
[ ] Carga por lotes de máximo 15.000 registros
[ ] MLflow con backend en PostgreSQL y artefactos en MinIO (sin SQLite)
[ ] La API usa el modelo que viene de MLflow
[ ] La API no tiene una ruta local quemada al modelo
[ ] Cada inferencia queda registrada en la base de datos
[ ] La API expone `/metrics`
[ ] Prometheus recolecta las métricas de la API
[ ] Grafana muestra el efecto de una prueba con Locust
[ ] Cada contenedor tiene `requests` y `limits`
[ ] Las imágenes propias están en DockerHub
