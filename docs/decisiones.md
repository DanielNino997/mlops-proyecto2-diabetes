# Decisiones de diseño

Carril B (datos, modelo y aplicación) y carril A (plataforma en Kubernetes).

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

## Riesgos conocidos (estado al cierre)
- ~~Los datos reales aún no se entrenaron en este entorno~~ → **Resuelto**: el DAG corrió 7 veces en el clúster (101.766 filas en `raw_data`, 99.340 en `clean_data`); `champion` = v7 (random_forest, test F2 = 0,4365).
- ~~Prometheus y réplicas~~ → **Resuelto**: Prometheus descubre cada pod de la API con `kubernetes_sd_configs` (ver sección de plataforma).

## Plataforma (carril A)

### Decisiones
- **Clúster**: k3s embebido en Rancher (contenedor `rancher/rancher` en la VM). Es el que usa el curso y la guía de Airflow indica no crear `kind` si ya hay un clúster. La VM no tiene sudo.
- **Namespaces**: `mlops` (datos, modelos, API, UI, observabilidad, Locust) y `airflow-local` (Airflow, como en la guía).
- **Un solo PostgreSQL** con dos bases: `mlops_data` (raw, clean, inference_logs, ingestion_batches) y `mlflow` (metadatos). Separa las capas sin gastar RAM en dos instancias. Airflow usa su propio PostgreSQL (el del chart).
- **Persistencia**: PVC (`local-path`) solo en PostgreSQL y MinIO, que guardan estado. MLflow, la API, Streamlit, Prometheus y Grafana no tienen estado propio (Grafana se provisiona desde ConfigMaps).
- **Versiones**: MLflow 2.16.2 (servidor y clientes), Airflow 3.0.6 (chart 1.22.0 con `airflowVersion: "3.0.6"`), Python 3.11. Imagen de MinIO `pgsty/minio` con versión fija (MinIO dejó de publicar imágenes oficiales).
- **Imágenes propias en DockerHub** con tag versionado (nunca `latest`): `mlops-p2-mlflow:2.16.2` y `mlops-p2-airflow:3.0.6-v1`. Cada cambio del DAG implica un tag nuevo.
- **ConfigMap y Secret por componente**: la configuración no sensible (hosts, puertos, URLs) va en ConfigMaps y las credenciales en Secrets, que los Deployments cargan con `envFrom`.
- **Variables de Airflow**: `k8s/airflow/config.yaml` crea un ConfigMap y un Secret en `airflow-local` con hosts completos (`postgres.mlops.svc.cluster.local`), porque un pod no puede leer Secrets de otro namespace ni resolver nombres cortos fuera del suyo.
- **Probes de la API**: readiness en `/ready` (solo recibe tráfico con el modelo cargado) y liveness en `/health`, con `timeoutSeconds: 5`. La liveness es más paciente (`failureThreshold: 6`) porque reiniciar es la medida más drástica.
- **Prometheus por pod**: `kubernetes_sd_configs` (role pod) con una ServiceAccount y un Role de solo lectura en `mlops`. Mide cada réplica de la API por separado y descubre réplicas nuevas sin cambiar configuración.
- **Grafana como código**: datasource y dashboard provisionados desde ConfigMaps (`monitoring/grafana/api-dashboard.json`); el tablero se recrea solo si el pod se reinicia.
- **Acceso**: el firewall de la VM solo permite SSH; las interfaces se exponen con `kubectl port-forward` + reenvío de puertos de VS Code. NodePort no aplica porque el clúster vive dentro del contenedor de Rancher.
- **Secretos**: están en el repositorio en texto plano (base64 no cifra). Aceptable para un ejercicio académico en una VM cerrada; en producción se usarían Sealed Secrets o un gestor de secretos.

### Resultados de las pruebas de carga
| Configuración | RPS promedio | Fallos | p50 | p95 | Reinicios |
|---|---:|---:|---:|---:|---:|
| 1 réplica (1 CPU / 1 GB), probes por defecto | 17,5 | 1,9 % | 2,8 s | 17 s | 5 |
| 3 réplicas (300m–1 CPU / 512Mi–1Gi), probes ajustadas | 48,4 | 0 % | 200 ms | 1,9 s | 0 |

Carga escalonada de 20 en 20 usuarios cada minuto hasta 200. Throughput ×2,8: no llega a 3× porque las réplicas comparten 4 CPU con Locust, PostgreSQL (que registra cada predicción), Prometheus y Airflow. La mejora de latencia es mayor que ×3 porque en la prueba de 1 réplica los reinicios por la liveness probe perdían la cola de peticiones. Detalle en `docs/pruebas_de_carga.md` y `docs/resultados/`.

### Dificultades encontradas y solución
1. **El k3s de Rancher no traía StorageClass** → ningún PVC se podía crear. Se instaló `local-path-provisioner` como StorageClass por defecto.
2. **`raw.githubusercontent.com` bloqueado desde la VM** → los archivos se trajeron con `git clone` por SSH y Helm desde `get.helm.sh`.
3. **Conflicto de dependencias**: Airflow 3.2.2 exige `cachetools>=6` y MLflow 2.16.2 `cachetools<6`. Se usó Airflow 3.0.6, compatible con MLflow 2.16.2 y con las versiones de scikit-learn de la API.
4. **Helm `--wait` + hooks**: las migraciones de Airflow son un hook que corre después de que todo esté listo, y los pods esperaban las migraciones (bloqueo mutuo). Solución: `useHelmHooks: false`.
5. **api-server de Airflow en reinicio continuo**: con 4 workers no cabía en 1 Gi y los procesos hijos morían. Solución: `AIRFLOW__API__WORKERS=1` y límite de 2 Gi.
6. **Probes por defecto bajo carga**: con 1 réplica saturada, `/health` tardaba más de 1 s y Kubernetes reinició la API 5 veces en plena prueba de Locust (falla en cascada). Con `timeoutSeconds: 5` hubo 0 reinicios en la prueba de 3 réplicas.
7. **Límite de 4 CPU de la VM**: 3 réplicas con `requests: 1 CPU` no cabían; se usaron `requests: 300m` y `limits: 1`.
8. **Sidecars `log-groomer` del chart sin `limits`** → se les agregaron recursos en `values.yaml` (condición de aceptación #12).
9. **`args` vs `command` en Locust**: `args` reemplazó el `CMD` completo de la imagen (incluido el ejecutable `locust`) y el pod quedó en `CrashLoopBackOff`. Solución: `command: ["locust"]` + `args`.