# Pruebas de carga con Locust

Sigue el método del **Taller Locust** (`annamerchanlm-dev/taller-locust`): medir la capacidad de la API de inferencia con
**1 réplica y con 3 réplicas** (mismos recursos por réplica), encontrar la concurrencia máxima que "soporta", y explicar qué recurso se satura primero.

## 1. Método (igual para 1 y 3 réplicas — no se cambia entre pruebas)

| Parámetro | Valor |
|---|---|
| Archivo | `locust/locustfile.py` (pacientes válidos aleatorios; `wait_time = between(1, 2.5)`, como el ejemplo del profe) |
| Éxito de una petición | HTTP 200 **y** respuesta con `prediction` y `probability` |
| Recursos por réplica | **1 CPU / 1 GB** (`requests = limits`) |
| Spawn rate | 10 usuarios/s |
| Ventana medida | 2 min **después** de crear todos los usuarios (`--reset-stats`); duración = usuarios/10 + 120 s |
| **Criterio "soporta"** | **< 1 % de fallos y p95 < 500 ms** en la ventana |
| Generador | **fuera del clúster/VM de la API** (otra máquina), para que Locust no compita por CPU. Vigilar que Locust no muestre `CPU usage above 90%` |
| RPS | columna `Requests/s` del `*_stats.csv` |
| CPU/RAM | promedio en la misma ventana (`kubectl top pod` o Grafana; ver §4) |

Se sube la carga por niveles (p. ej. 1 réplica: 50, 80, 100, 110, 120, 130, 140; 3 réplicas: 150, 300, 400, 450, 500, 550) hasta que un nivel deje de cumplir. **Concurrencia máxima sostenible** = el último nivel que cumple. En el taller: 1 réplica ≈ 110 usuarios / 54 RPS y 3 réplicas ≈ 480 usuarios / 230 RPS (otra API más liviana; la nuestra tendrá otros números).

## 2. Ejecución

Preparar la API (una sola vez por configuración) y confirmar que todas las réplicas están listas **antes** de medir:

```bash
kubectl set resources deployment/inference-api --requests=cpu=1,memory=1Gi --limits=cpu=1,memory=1Gi
kubectl scale deployment/inference-api --replicas=1        # luego --replicas=3
kubectl rollout status deployment/inference-api
kubectl get pods -l app=inference-api                       # todas Ready (ajustar la etiqueta a la real)
curl <url-api>/ready
```

Un nivel (en la máquina generadora; ejemplo 110 usuarios → 110/10 + 120 = 131 s):

```bash
locust -f locust/locustfile.py --host http://<url-api> --headless \
       -u 110 -r 10 -t 131s --reset-stats \
       --csv resultados/rep1_u110 --html resultados/rep1_u110.html
```

Con Docker (como en el taller): `docker run --rm -v $PWD:/mnt/locust locustio/locust:2.32.5 -f /mnt/locust/locust/locustfile.py ...` o la imagen `dnino97/locust-diabetes` (`locust/Dockerfile`). Para ver los gráficos en vivo (evidencia), abrir la interfaz web (`locust -f ... --host ...` y entrar a :8089) y tomar capturas de *Statistics*, *Charts* y *Failures*.

Opcional, para hallar el límite de una vez: `LOAD_SHAPE=steps STEP_USERS=20 STEP_SECONDS=60 STEP_MAX_USERS=200`. El resultado "oficial" se mide con niveles fijos como arriba. `INVALID_RATIO=0.2` mezcla solicitudes inválidas (422 esperado, no cuenta como fallo): usarlo solo para una prueba aparte.

## 3. Tabla de resultados (llenar por nivel)

**1 réplica (1 CPU / 1 GB)**

| Usuarios | RPS | Peticiones | Fallos | p50 | p95 | p99 | CPU réplica | RAM réplica | Cumple |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| | | | | | | | | | |

**3 réplicas (1 CPU / 1 GB cada una)**

| Usuarios | RPS | Peticiones | Fallos | p50 | p95 | p99 | CPU por réplica | CPU total API | RAM por réplica | Cumple |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| | | | | | | | | | | |

## 4. Preguntas que hay que responder (las del taller)

1. Concurrencia máxima sostenible y RPS con **1 réplica**.
2. Concurrencia máxima sostenible y RPS con **3 réplicas**.
3. ¿En qué % cambió la capacidad? ¿Fue cercano a 3×? (calcular usuarios, RPS en el límite y techo de RPS; si no es 3×, explicar por qué).
4. ¿Qué recurso se saturó primero y qué evidencia hay? (CPU de la réplica al ~100 % con RPS plano; RAM; el generador; la red).
5. ¿Qué servicios adicionales pudieron limitar el resultado? (balanceador/Ingress, PostgreSQL por el registro en `inference_logs`, MLflow solo al arrancar, nodo compartido con Airflow/Prometheus).

**Medir CPU/RAM:** `kubectl top pod -l app=inference-api` cada pocos segundos durante la ventana (script: `while true; do date +%s; kubectl top pod -l app=inference-api --no-headers; sleep 5; done >> resultados/rep1_u110_top.txt`) o los paneles de CPU/memoria de Grafana (`docs/metricas.md`).

## 5. Particularidades de nuestro proyecto (no estaban en el taller)
- **PostgreSQL entra en la ruta de cada petición**: cada predicción se registra en `inference_logs` (en segundo plano, sin sumar latencia a la respuesta, pero consumiendo CPU y conexiones). Vigilar `diabetes_api_inference_log_writes_total{result!="ok"}` y la CPU de Postgres durante la prueba; es un candidato a cuello de botella con 3 réplicas.
- **Balanceo**: el Service de Kubernetes reparte por **conexión**. Locust mantiene una conexión por usuario, así que con cientos de usuarios queda parejo; con pocos usuarios puede cargar más a una réplica. Revisar en Grafana (`sum by (pod)`) que las 3 réplicas reciban tráfico similar. Si hay Ingress/Nginx delante, revisar `max_fails`, `fail_timeout` y `keepalive_timeout` (el taller perdió capacidad por la configuración por defecto de Nginx: 502 `no live upstreams`).
- **Un proceso por pod** (uvicorn sin `--workers`): escalar es con réplicas.
- Cada predicción deja una fila en `inference_logs`: una barrida completa genera decenas de miles. Limpiar después solo lo de Locust: `DELETE FROM inference_logs WHERE source='locust';`
- No entrenar (DAG) mientras se mide: compite por CPU del nodo y falsea el resultado.
- Repetir un nivel dudoso al menos dos veces; la primera corrida puede ser más lenta (arranque en frío).
- Evidencia (nombres sugeridos): `rep1_u110_charts.png`, `rep1_u110_stats.png`, `rep3_u480_charts.png`, etc., más los CSV/HTML de Locust, `kubectl get pods` y el tablero de Grafana de cada nivel. Ver `docs/evidencias.md`.
