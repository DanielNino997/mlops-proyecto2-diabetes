# Lista de pantallazos y evidencias

Guardar en `docs/evidencias/` con el nombre indicado. Marcar con ✅ al tomarlo. Hacerlos **con datos reales corriendo**, con fecha/hora visible si es posible.

## Datos y Airflow
- [ ] `01_airflow_dag_grafo.png` — vista *Graph* del DAG `diabetes_pipeline` (todas las tareas).
- [ ] `02_airflow_ejecucion_ok.png` — una ejecución en verde (lote 1) con duración por tarea.
- [ ] `03_airflow_ejecucion_lote2.png` — segunda ejecución (lote 2) para mostrar el avance por lotes.
- [ ] `04_airflow_archivo_agotado.png` — ejecución con tareas *skipped* cuando ya no hay filas.
- [ ] `05_airflow_log_train.png` — log de una tarea de entrenamiento.
- [ ] `06_bd_ingestion_batches.png` — `SELECT * FROM ingestion_batches;`
- [ ] `07_bd_conteos.png` — `SELECT 'raw', count(*) FROM raw_data UNION ALL SELECT 'clean', count(*) FROM clean_data;` y `SELECT split, count(*) FROM clean_data GROUP BY 1;`

## MLflow
- [ ] `08_mlflow_experimentos.png` — lista de corridas de los 3 modelos con métricas.
- [ ] `09_mlflow_comparacion.png` — comparación de corridas (F2, recall, precisión, PR-AUC).
- [ ] `10_mlflow_corrida_detalle.png` — parámetros, métricas y artefactos de una corrida (matriz de confusión, curva PR).
- [ ] `11_mlflow_registry.png` — Model Registry con versiones y alias `champion`.
- [ ] `12_mlflow_etiquetas_version.png` — etiquetas de la versión promovida (umbral, razón).
- [ ] `13_minio_artefactos.png` — bucket de MinIO con los artefactos.

## API
- [ ] `14_api_docs.png` — `/docs` (Swagger) con los endpoints.
- [ ] `15_api_predict.png` — `POST /predict` exitoso (respuesta completa).
- [ ] `16_api_predict_422.png` — `POST /predict` con dato inválido (422).
- [ ] `17_api_model_info.png` — `/model-info`.
- [ ] `18_api_metrics.png` — `/metrics`.
- [ ] `19_bd_inference_logs.png` — `SELECT request_id, status, prediction, probability, model_version, source, latency_ms FROM inference_logs ORDER BY id DESC LIMIT 10;`
- [ ] `20_api_cambio_champion.png` — antes/después: `/model-info` muestra la versión nueva tras promover, **sin reiniciar** la API.

## Streamlit
- [ ] `21_streamlit_formulario.png` — formulario con valores de ejemplo cargados.
- [ ] `22_streamlit_prediccion.png` — resultado: probabilidad, versión del modelo.
- [ ] `23_streamlit_error_validacion.png` — error de validación legible (pestaña JSON manual).

## Kubernetes y observabilidad
- [ ] `24_k8s_pods.png` — `kubectl get pods,svc` con todo en *Running*.
- [ ] `25_k8s_deployment_api.png` — Deployment con réplicas y límites de recursos.
- [ ] `26_prometheus_targets.png` — *Targets* de la API en estado UP.
- [ ] `27_grafana_dashboard.png` — tablero completo: total, RPS, latencia promedio/p50/p95/p99, errores, CPU, memoria.

## Pruebas de carga (ver `pruebas_de_carga.md`)
Por cada experimento A / B / C (1, 2 y 4 réplicas):
- [ ] `28_locust_A_estadisticas.png`, `29_locust_A_graficas.png`, `30_grafana_A.png`
- [ ] `31_locust_B_estadisticas.png`, `32_locust_B_graficas.png`, `33_grafana_B.png`
- [ ] `34_locust_C_estadisticas.png`, `35_locust_C_graficas.png`, `36_grafana_C.png`
- [ ] `37_bd_logs_locust.png` — `SELECT source, status, count(*) FROM inference_logs GROUP BY 1,2;`

## Repositorio
- [ ] `38_github_estructura.png` — árbol del repositorio.
- [ ] `39_dockerhub_imagenes.png` — imágenes publicadas.
