# MLOps Proyecto 2: readmisión de pacientes con diabetes

Proyecto 2 (Nivel 3) del curso **Operaciones de Machine Learning**, Pontificia Universidad Javeriana.

Sistema MLOps completo desplegado en Kubernetes: los datos clínicos llegan por lotes, se guardan crudos, se limpian, entrenan un modelo que se registra en MLflow, y ese modelo se sirve por una API que se consulta desde una interfaz web. Todo con pruebas de carga y observabilidad.

**Problema:** predecir si un paciente con diabetes será readmitido en el hospital en menos de 30 días (clasificación binaria).

**Dataset:** Diabetes 130-US hospitals (1999-2008), versión modificada entregada por el curso.

> **Estado:** en construcción. Entrega: 2 de noviembre de 2026.

## Equipo

| Carril | Responsable | Componentes |
|---|---|---|
| A · Plataforma | Ana María Merchán León | Kubernetes, PostgreSQL, MinIO, MLflow, Airflow, Prometheus, Grafana, imágenes en DockerHub |
| B · ML y aplicación | Daniel Niño | EDA, preprocesamiento, entrenamiento, API FastAPI, Streamlit, Locust |
| Conjunto | Ambos | Contratos entre componentes, DAG de Airflow, pruebas de carga, documentación y video |

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
| `docs/` | Registro de decisiones del proyecto (`decisiones.md`) |

## Contratos entre componentes

Para que las piezas encajen, la forma en que se comunican está escrita una sola vez en el código y todos la importan:

| Archivo | Qué define |
|---|---|
| `src/config.py` | Nombre del modelo en MLflow, alias `champion`, nombres de tablas y Services. Todo se lee de variables de entorno (ConfigMap y Secret en Kubernetes). |
| `src/schemas.py` | Datos que recibe y devuelve `/predict`, respuestas de `/model-info` y `/health`, lista de
