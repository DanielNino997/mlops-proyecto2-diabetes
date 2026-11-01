# Métricas de la API para Prometheus / Grafana

La API expone `GET /metrics` (formato Prometheus). Todas las métricas empiezan por `diabetes_api_`.
Prometheus debe hacer *scrape* a cada pod de `inference-api` en el puerto 8000, ruta `/metrics`
(con réplicas, hay que descubrir los pods —por Service/endpoints o `kubernetes_sd_configs`—, no solo el Service).

## Métricas disponibles

| Métrica | Tipo | Etiquetas | Qué mide |
|---|---|---|---|
| `diabetes_api_requests_total` | counter | `method`, `endpoint`, `status` | Solicitudes HTTP atendidas |
| `diabetes_api_request_duration_seconds` | histogram | `endpoint` | Duración de cada solicitud (de ahí salen p50/p95/p99) |
| `diabetes_api_requests_in_progress` | gauge | — | Solicitudes en curso ahora mismo |
| `diabetes_api_predictions_total` | counter | `prediction` (0/1) | Predicciones entregadas por clase |
| `diabetes_api_prediction_probability` | histogram | — | Distribución de la probabilidad (vigilar deriva) |
| `diabetes_api_inference_duration_seconds` | histogram | — | Tiempo solo del modelo (sin red ni BD) |
| `diabetes_api_prediction_errors_total` | counter | `reason` | Fallos al predecir |
| `diabetes_api_inference_log_writes_total` | counter | `result` | Escrituras a `inference_logs` (ok / error) |
| `diabetes_api_model_loaded` | gauge | — | 1 si hay modelo cargado |
| `diabetes_api_model_version` | gauge | — | Versión del modelo en uso |
| `diabetes_api_model_info` | gauge | `model_name`, `model_version`, `model_alias` | Modelo en uso (valor 1) |
| `diabetes_api_model_loaded_timestamp_seconds` | gauge | — | Cuándo se cargó el modelo |
| `diabetes_api_model_reloads_total` | counter | `result` | Recargas del modelo |

`endpoint` usa la **plantilla** de la ruta (`/predict`), nunca valores variables: no hay explosión de series.
Los buckets del histograma de duración van de 5 ms a 10 s.

## Paneles que pide el enunciado (consultas PromQL)

| Panel | PromQL |
|---|---|
| Total de solicitudes (`/predict`) | `sum(diabetes_api_requests_total{endpoint="/predict"})` |
| Solicitudes por segundo | `sum(rate(diabetes_api_requests_total{endpoint="/predict"}[1m]))` |
| Latencia promedio | `sum(rate(diabetes_api_request_duration_seconds_sum{endpoint="/predict"}[1m])) / sum(rate(diabetes_api_request_duration_seconds_count{endpoint="/predict"}[1m]))` |
| Latencia p50 | `histogram_quantile(0.50, sum by (le) (rate(diabetes_api_request_duration_seconds_bucket{endpoint="/predict"}[1m])))` |
| Latencia p95 | igual con `0.95` |
| Latencia p99 | igual con `0.99` |
| Errores del servidor (5xx) por segundo | `sum(rate(diabetes_api_requests_total{endpoint="/predict",status=~"5.."}[1m]))` |
| Tasa de error 5xx | `sum(rate(diabetes_api_requests_total{endpoint="/predict",status=~"5.."}[1m])) / sum(rate(diabetes_api_requests_total{endpoint="/predict"}[1m]))` |
| Solicitudes rechazadas por validación (422) | `sum(rate(diabetes_api_requests_total{endpoint="/predict",status="422"}[1m]))` |
| Solicitudes en curso | `sum(diabetes_api_requests_in_progress)` |
| Predicciones por clase | `sum by (prediction) (rate(diabetes_api_predictions_total[5m]))` |
| Versión del modelo en uso | `max(diabetes_api_model_version)` |
| Fallos al escribir `inference_logs` | `sum(rate(diabetes_api_inference_log_writes_total{result!="ok"}[5m]))` |

Si la API corre con varias réplicas, estas consultas ya suman entre pods (`sum`). Para ver cada pod por separado,
cambiar `sum(...)` por `sum by (pod) (...)` (la etiqueta `pod` la agrega Prometheus al hacer scrape).

## CPU y memoria (no los expone la API)

Salen de cAdvisor / kubelet (ya incluidos con kube-prometheus-stack). Ajustar el nombre del pod si es distinto:

| Panel | PromQL |
|---|---|
| CPU por pod (cores) | `sum by (pod) (rate(container_cpu_usage_seconds_total{pod=~"inference-api.*",container!=""}[1m]))` |
| Memoria por pod | `sum by (pod) (container_memory_working_set_bytes{pod=~"inference-api.*",container!=""})` |
| CPU vs límite (%) | `sum by (pod) (rate(container_cpu_usage_seconds_total{pod=~"inference-api.*",container!=""}[1m])) / sum by (pod) (kube_pod_container_resource_limits{resource="cpu",pod=~"inference-api.*"})` |

## Comprobación rápida

```bash
kubectl port-forward svc/inference-api 8000:8000
curl -s localhost:8000/metrics | grep ^diabetes_api_requests_total
```
