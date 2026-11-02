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

## 3. Resultados (prueba escalonada, Locust dentro del clúster)

**Cómo se midió realmente.** Se usó la variante escalonada (`LOAD_SHAPE=steps`, +20 usuarios cada 60 s hasta 200) en lugar de niveles fijos de 2 min, porque encontraba el límite en una sola corrida. Cada fila toma la segunda mitad del escalón (ya estabilizado) de la serie temporal del reporte HTML de Locust (`docs/resultados/rep*_steps_report.html`); RPS y percentiles son los de la ventana móvil de Locust. Criterio "soporta": < 1 % de fallos y p95 < 500 ms. La granularidad es de 20 usuarios, así que el límite real está entre el último nivel que cumple y el siguiente.

**Diferencias con el método del taller (declaradas):** Locust corrió dentro del mismo clúster (VM de 4 CPU compartida con PostgreSQL, Prometheus y Airflow), no desde otra máquina; con 1 réplica las probes eran las de por defecto y con 3 réplicas se usaron `requests 300m / limits 1 CPU` y probes con timeout de 5 s (con `requests 1 CPU` las 3 réplicas no cabían en la VM).

**1 réplica (1 CPU / 1 GB, probes por defecto)**

| Usuarios | RPS | p50 | p95 | Fallos/s (máx.) | Cumple |
|---:|---:|---:|---:|---:|:---:|
| 20 | 11,1 | 49 ms | 303 ms | 0 | Sí |
| 40 | 20,0 | 200 ms | 688 ms | 0 | No (p95) |
| 60 | 21,7 | 1.030 ms | 1.883 ms | 0 | No |
| 80 | 22,0 | 2.017 ms | 3.067 ms | 0 | No |
| 100 | 21,5 | 2.933 ms | 3.900 ms | 0 | No |
| 120 | 18,0 | 5.450 ms | 21.667 ms | 0,4 | No |
| 140 | 12,8 | 7.583 ms | 16.267 ms | 1,0 | No |
| 160 | 15,3 | 8.050 ms | 8.867 ms | 1,4 | No |
| 180 | 21,1 | 6.917 ms | 7.617 ms | 1,4 | No |
| 200 | 17,5 | 7.467 ms | 34.667 ms | 0 | No |

**3 réplicas (300m–1 CPU / 512Mi–1Gi cada una, probes con timeout 5 s)**

| Usuarios | RPS | p50 | p95 | Fallos/s (máx.) | Cumple |
|---:|---:|---:|---:|---:|:---:|
| 20 | 11,2 | 37 ms | 82 ms | 0 | Sí |
| 40 | 22,2 | 41 ms | 123 ms | 0 | Sí |
| 60 | 33,1 | 42 ms | 132 ms | 0 | Sí |
| 80 | 44,0 | 56 ms | 218 ms | 0 | Sí |
| 100 | 54,2 | 75 ms | 332 ms | 0 | Sí |
| 120 | 61,8 | 115 ms | 573 ms | 0 | No (p95) |
| 140 | 67,8 | 228 ms | 865 ms | 0 | No |
| 160 | 66,5 | 563 ms | 1.583 ms | 0 | No |
| 180 | 67,0 | 935 ms | 1.817 ms | 0 | No |
| 200 | 65,7 | 1.317 ms | 2.350 ms | 0 | No |

**Totales de toda la prueba** (`docs/resultados/rep*_steps_requests.csv`): 1 réplica, 10.382 peticiones, 197 fallos (1,9 %), 17,5 RPS promedio, p95 17 s, 5 reinicios del pod; 3 réplicas, 29.098 peticiones, 0 fallos, 48,4 RPS promedio, p95 1,9 s, 0 reinicios.

## 4. Preguntas que hay que responder (las del taller)

1. Concurrencia máxima sostenible y RPS con **1 réplica**.
2. Concurrencia máxima sostenible y RPS con **3 réplicas**.
3. ¿En qué % cambió la capacidad? ¿Fue cercano a 3×? (calcular usuarios, RPS en el límite y techo de RPS; si no es 3×, explicar por qué).
4. ¿Qué recurso se saturó primero y qué evidencia hay? (CPU de la réplica al ~100 % con RPS plano; RAM; el generador; la red).
5. ¿Qué servicios adicionales pudieron limitar el resultado? (balanceador/Ingress, PostgreSQL por el registro en `inference_logs`, MLflow solo al arrancar, nodo compartido con Airflow/Prometheus).

**Respuestas** (con los datos de la sección 3)

1. **1 réplica:** la máxima concurrencia que cumple el criterio es **20 usuarios (~11 RPS)**; el límite real está entre 20 y 40 usuarios (a 40 el p95 ya es 688 ms). El techo de RPS es de ~22 y se alcanza desde ~60 usuarios.
2. **3 réplicas:** **100 usuarios (~54 RPS)**; el límite real está entre 100 y 120 (a 120 el p95 es 573 ms). El techo de RPS es de ~67 desde ~140 usuarios.
3. **Cambio de capacidad:** en usuarios sostenibles pasa de 20 a 100 (×5) y en RPS en el límite de 11 a 54 (×4,9), pero esa comparación está inflada por la granularidad de 20 usuarios (el límite de 1 réplica está entre 20 y 40). La medida más fiable es el **techo de RPS: ~22 → ~67, es decir ×3,0**, prácticamente lineal (~22 RPS por CPU). Sobre el promedio de toda la prueba el aumento es ×2,8 (17,5 → 48,4 RPS) porque la prueba de 1 réplica incluye el periodo con reinicios del pod.
4. **Recurso que se satura primero: la CPU de la réplica de la API** (un proceso uvicorn por pod, límite de 1 CPU). Evidencia: con 1 réplica el RPS se aplana en ~22 desde 60 usuarios mientras la latencia sigue creciendo, y con 3 réplicas se aplana en ~67 (×3). Pendiente de confirmar con la captura de CPU por pod en Grafana (`rate(container_cpu_usage_seconds_total)` o `kubectl top pod`), que no está en los CSV de Locust. A partir de ~120 usuarios con 1 réplica la liveness probe (timeout 1 s) reinició el pod 5 veces (fallos `Connection refused`/`reset`), lo que explica la caída de RPS a 13–18.
5. **Servicios adicionales que pudieron limitar:** los 4 CPU de la VM se comparten con Locust, PostgreSQL (que recibe un `INSERT` por predicción en `inference_logs`), Prometheus y Airflow; por eso 3 réplicas no superan ×3. Locust llama al Service de Kubernetes (`inference-api:8000`) sin Ingress ni Nginx intermedio. MLflow solo se consulta al arrancar y cada 60 s para ver el alias, no por petición.

**Medir CPU/RAM:** `kubectl top pod -l app=inference-api` cada pocos segundos durante la ventana (script: `while true; do date +%s; kubectl top pod -l app=inference-api --no-headers; sleep 5; done >> resultados/rep1_u110_top.txt`) o los paneles de CPU/memoria de Grafana (`docs/metricas.md`).

## 5. Particularidades de nuestro proyecto (no estaban en el taller)
- **PostgreSQL entra en la ruta de cada petición**: cada predicción se registra en `inference_logs` (en segundo plano, sin sumar latencia a la respuesta, pero consumiendo CPU y conexiones). Vigilar `diabetes_api_inference_log_writes_total{result!="ok"}` y la CPU de Postgres durante la prueba; es un candidato a cuello de botella con 3 réplicas.
- **Balanceo**: el Service de Kubernetes reparte por **conexión**. Locust mantiene una conexión por usuario, así que con cientos de usuarios queda parejo; con pocos usuarios puede cargar más a una réplica. Revisar en Grafana (`sum by (pod)`) que las 3 réplicas reciban tráfico similar. Si hay Ingress/Nginx delante, revisar `max_fails`, `fail_timeout` y `keepalive_timeout` (el taller perdió capacidad por la configuración por defecto de Nginx: 502 `no live upstreams`).
- **Un proceso por pod** (uvicorn sin `--workers`): escalar es con réplicas.
- Cada predicción deja una fila en `inference_logs`: una barrida completa genera decenas de miles. Limpiar después solo lo de Locust: `DELETE FROM inference_logs WHERE source='locust';`
- No entrenar (DAG) mientras se mide: compite por CPU del nodo y falsea el resultado.
- Repetir un nivel dudoso al menos dos veces; la primera corrida puede ser más lenta (arranque en frío).
- Evidencia (nombres sugeridos): `rep1_u110_charts.png`, `rep1_u110_stats.png`, `rep3_u480_charts.png`, etc., más los CSV/HTML de Locust, `kubectl get pods` y el tablero de Grafana de cada nivel. Ver `docs/evidencias.md`.
