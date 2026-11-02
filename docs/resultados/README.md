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

- Techo de RPS: ~22 con 1 réplica (desde ~60 usuarios) y ~67 con 3 réplicas (desde ~140). Con el criterio <1 % de fallos y p95 <500 ms, 1 réplica sostiene 20 usuarios y 3 réplicas 100.
- Los fallos de 1 réplica (`Connection refused`/`reset`, `HTTP 0`) se deben a que la liveness probe reinició el pod bajo carga.
- El techo de RPS crece ×3,0 (~22 → ~67). El promedio de toda la prueba sube ×2,8 porque la de 1 réplica incluye los reinicios del pod. CPU por pod: con 1 réplica llega a ~0,93 núcleos (se satura la réplica); con 3 réplicas cada pod se queda en ~0,67 (el límite pasa a ser la CPU compartida de la VM). Ver `docs/evidencias/30_grafana_cpu_por_pod.png` y `cpu_por_pod.txt`.
