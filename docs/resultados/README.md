# Resultados de las pruebas de carga (Locust)

Carga escalonada: 20 usuarios más cada 60 s hasta 200 (`LOAD_SHAPE=steps`), `wait_time = between(1, 2.5)`.
Locust corre dentro del clúster contra `http://inference-api:8000`.

| Archivo | Configuración |
|---|---|
| `rep1_steps_*` | 1 réplica, 1 CPU / 1 GB, probes por defecto (liveness timeout 1 s) |
| `rep3_steps_*` | 3 réplicas, 300m-1 CPU / 512Mi-1Gi, probes con timeout 5 s |

| Métrica | 1 réplica | 3 réplicas |
|---|---:|---:|
| Peticiones | 10.382 | 29.098 |
| Fallos | 197 (1,9 %) | 0 (0 %) |
| RPS promedio | 17,5 | 48,4 |
| p50 | 2.800 ms | 200 ms |
| p95 | 17.000 ms | 1.900 ms |
| p99 | 34.000 ms | 2.500 ms |
| Máximo | 72.000 ms | 3.700 ms |
| Reinicios de la API | 5 | 0 |

- Con 1 réplica el RPS se aplana en ~18 req/s hacia los 40 usuarios y la latencia se dispara.
- Los fallos de 1 réplica (`Connection refused`/`reset`, `HTTP 0`) se deben a que la liveness probe reinició el pod bajo carga.
- Throughput ×2,8 con 3 réplicas: no llega a 3× porque las réplicas comparten 4 CPU con Locust, PostgreSQL, Prometheus y Airflow.
