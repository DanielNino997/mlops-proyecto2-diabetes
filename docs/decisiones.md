# Decisiones de diseño (carril B: datos, modelo y aplicación)

Formato: **decisión → por qué → alternativa descartada**.

## Datos
1. **Carga por lotes de 15.000 filas** (`BATCH_SIZE`), un lote por ejecución del DAG. → El enunciado pide simular llegada de datos
   en el tiempo. *Descartado:* cargar todo de una vez.
2. **Idempotencia** (`row_hash` único + `ON CONFLICT DO NOTHING`, tabla `ingestion_batches` por `dag_run_id`). → Un reintento o una
   ejecución repetida no duplica filas; un lote que falló se **retoma**, no se salta. *Descartado:* truncar y recargar.
3. **`raw_data` guarda todo como texto, sin transformar**. → Es la copia fiel de la fuente; los errores de limpieza se pueden rehacer sin volver al CSV.
4. **Limpieza en `src/preprocessing.py`** (una sola función usada por el DAG, el EDA y las pruebas). Decisiones sobre el dataset:
   - Se eliminan encuentros de pacientes fallecidos o en hospicio (`discharge_disposition_id` ∈ {11,13,14,19,20,21}): no pueden reingresar y ensucian la etiqueta.
   - `?` y vacío se tratan como nulo; el texto `None` **no** es nulo: en `A1Cresult` y `max_glu_serum` significa "prueba no realizada" y es una categoría válida.
   - Se eliminan encuentros repetidos (mismo `encounter_id`) y los de `gender` inválido.
   - `race` nulo → `Unknown`; diagnóstico ausente → `Missing` (no se imputa).
   - Diagnósticos ICD-9 se agrupan en 9 familias + `Missing` (Strack et al., 2014); la especialidad médica (73 valores) en 7 grupos (las poco frecuentes → `Other`).
   - Solo se conservan las columnas del contrato de la API (24 variables + identificadores + objetivo). Se descartan, entre otras, `weight` y `payer_code` (muy incompletas) y el resto de columnas de medicamentos.
5. **Objetivo binario**: `readmitted == "<30"` → 1; `">30"` y `"NO"` → 0 (columna `readmitted_lt30`). Es lo que pide el negocio (readmisión temprana).
6. **Partición por paciente, no por fila** (hash MD5 de `patient_nbr` con sal fija → 70/15/15 train/val/test, guardada en `clean_data.split`).
   → Un mismo paciente tiene varios encuentros; si se mezclaran filas, el modelo vería al paciente en entrenamiento y en prueba (fuga de datos).
   La partición es determinista: un paciente que reaparece en un lote posterior cae siempre en el mismo conjunto.
   > Cambio respecto a lo acordado: `clean_data` guarda además `batch_id`, `row_hash`, `split`, `processed_at`.

## Modelo
7. **Pipeline de sklearn** (escalado + one-hot **dentro** del modelo registrado). → La API recibe valores crudos y nunca repite transformaciones; evita desfase entre entrenamiento y servicio.
8. **Tres candidatos** (regresión logística, random forest, histogram gradient boosting), cada uno es una corrida de MLflow con parámetros, métricas, matriz de confusión y curva PR.
9. **Métrica principal: F2** (recall pesa el doble que precisión). → En readmisión temprana es peor **no detectar** a un paciente de riesgo que alertar de más.
   El **umbral de decisión se optimiza en validación** (no se usa 0,5) y se guarda con el modelo.
   *Consecuencia que hay que explicar:* con F2 el modelo prefiere recall alto y precisión baja; la clase positiva es ~11 % del total, así que
   muchas alertas son falsos positivos. Es una decisión de negocio, no un error. Se reportan también precisión, recall, F1, PR-AUC.
10. **Selección y promoción sin trampa**: el mejor candidato se elige con **validación**; el champion actual se **re-evalúa en el mismo conjunto de prueba**
    antes de compararlos (con verificación de huella del conjunto). Solo se promueve si **supera** al champion; en empate se conserva el champion.
    La decisión queda como etiquetas en la versión del registro (umbral, promoted, razón).
11. **Alias `champion`** del Model Registry de MLflow (no etapas, que están obsoletas en MLflow 3).
12. **Mínimo de positivos por partición (30)**: si un lote no deja suficientes casos positivos en validación/prueba, la tarea falla con un mensaje claro en vez de producir métricas sin sentido.

## API
13. **Carga del modelo por alias y refresco en segundo plano** (cada `MODEL_CACHE_TTL_SECONDS`=60 s, solo recarga si cambió la versión).
    `/predict` **nunca llama a MLflow**: si MLflow cae, la API sigue sirviendo el último modelo bueno. Al promover un champion nuevo no hace falta reiniciar nada.
14. **Validación estricta con Pydantic** (`extra="forbid"`, `Literal` para categorías, rangos para números): una entrada inválida devuelve 422 con el detalle por campo.
15. **Registro de inferencias en segundo plano** (`BackgroundTasks` → `inference_logs`, con `request_id`, entrada JSONB, predicción, probabilidad, versión del modelo, latencia, `source`).
    → El registro no suma latencia a la respuesta; un fallo de BD no tumba la predicción (se cuenta en `diabetes_api_inference_log_writes_total`).
16. **Un proceso por pod**; se escala con réplicas de Kubernetes. Las métricas son de Prometheus con histogramas (percentiles) y etiquetas por **plantilla de ruta** (sin cardinalidad infinita).
17. Endpoints: `/health` (vivo), `/ready` (modelo cargado → readiness probe), `/model-info`, `/predict`, `/metrics`.

## Interfaz y pruebas de carga
18. **Streamlit solo habla con la API** (no importa MLflow ni se conecta a la BD). Genera el formulario desde el contrato (`schemas.py`), tiene valores de ejemplo y una pestaña de JSON manual para demostrar el rechazo de datos inválidos.
19. **Locust** genera pacientes válidos aleatorios desde el mismo contrato y marca `X-Client: locust`; las solicitudes inválidas intencionales esperan 422.

## Compatibilidad con la plataforma (carril A)
- **MLflow**: el servidor es **2.16.2** (Postgres + MinIO, `--no-serve-artifacts`). El cliente de la API y del DAG se fija en **`mlflow-skinny==2.16.2`**: un cliente 3.x puede fallar contra un servidor 2.x al registrar modelos. Probado contra un servidor 2.16.2 real con artefactos en S3 (entrenar, registrar, promover y servir).
- **Airflow 3.0.6** (Helm chart oficial). Trae SQLAlchemy 1.4, por eso `db.py` no usa `pandas.read_sql_query` (pandas 2.2+ exige SQLAlchemy 2). DAG probado de punta a punta en Airflow 3.0.6: lote 1, lote 2 y archivo agotado (tareas *skipped*).
- **Dependencias del DAG** (`airflow/requirements.txt`): se instalan **sin** las restricciones de Airflow porque fijan scikit-learn 1.7 y el modelo debe entrenarse con la **misma versión que la API** (1.6.1, junto con numpy 2.0.2 y pandas 2.3.3). `pip check` verifica que no se rompa Airflow.
- **Esquema de tablas**: se usa el `sql/init.sql` de Ana sin cambios (`raw_data` guarda cada fila completa en `payload` JSONB). `sql/pipeline.sql` agrega lo que falta y el DAG lo ejecuta en `init_schema`: la tabla `ingestion_batches` (control de lotes) y la columna `clean_data.split`.
- **Base de datos de datos**: `mlops_data` (la de MLflow es otra: `mlflow`), ambas en el mismo PostgreSQL.

## Riesgos conocidos
- **Los datos reales aún no se entrenaron en este entorno de desarrollo**: las métricas finales salen de la primera ejecución del DAG en el clúster (registrarlas en la presentación).
- **Prometheus y réplicas**: scrapear el Service `inference-api` reparte cada lectura entre pods distintos y mezcla los contadores. Con varias réplicas hay que descubrir los pods (Service headless o `kubernetes_sd_configs`).
