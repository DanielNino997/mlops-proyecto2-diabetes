# Imagen de Airflow con el DAG, el código del proyecto y el CSV.
# Contexto de build = raíz del repositorio (el CSV debe estar en data/Diabetes/Diabetes.csv):
#   docker build -f airflow/Dockerfile -t dnino97/airflow-diabetes:3.0.6 .
# Debe usar la MISMA versión de Airflow que el Helm chart de k8s/airflow
# (probado con 3.0.6). Se la indicamos al chart con images.airflow.repository/tag.
ARG AIRFLOW_VERSION=3.0.6
FROM apache/airflow:${AIRFLOW_VERSION}

COPY airflow/requirements.txt /tmp/requirements.txt
# Sin --constraint de Airflow: fija scikit-learn 1.7 y aquí necesitamos 1.6.1,
# igual que la API (el modelo entrenado aquí se lee allá). Verificado con pip check.
RUN pip install --no-cache-dir -r /tmp/requirements.txt && pip check

# Código compartido (config, db, preprocessing, train), esquema SQL, DAG y datos
COPY --chown=airflow:root src/ /opt/airflow/src/
COPY --chown=airflow:root sql/ /opt/airflow/sql/
COPY --chown=airflow:root airflow/dags/ /opt/airflow/dags/
COPY --chown=airflow:root data/Diabetes/Diabetes.csv /opt/airflow/data/Diabetes/Diabetes.csv

ENV PROJECT_SRC=/opt/airflow/src SQL_DIR=/opt/airflow/sql
