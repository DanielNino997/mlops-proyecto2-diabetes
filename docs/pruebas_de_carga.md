# Pruebas de carga de la API de inferencia

Se midió cuánta carga soporta la API (`POST /predict`) con **1 réplica y con 3 réplicas**, siguiendo el criterio del Taller Locust: una configuración "soporta" un nivel de usuarios si hay **menos de 1 % de fallos y p95 < 500 ms**.

## Conclusión

- Con **1 réplica** la API sostiene **20 usuarios (~11 RPS)**; su techo de throughput es ~22 RPS y bajo más carga se degrada y el pod se reinicia.
- Con **3 réplicas** sostiene **100 usuarios (~54 RPS)**, con 0 fallos y 0 reinicios en toda la prueba; su techo es ~67 RPS.
- El techo de RPS crece ×3,0 (~22 → ~67): el escalado es prácticamente lineal. El recurso que se satura primero es la CPU de cada réplica.

## 1. Configuración de la prueba

| Parámetro | Valor |
|---|---|
| Escenario | `locust/locustfile.py`: pacientes válidos aleatorios contra `POST /predict` (y `GET /model-info`), `wait_time = between(1, 2.5)` |
| Éxito de una petición | HTTP 200 y respuesta con `prediction` y `probability` |
| Perfil de carga | Escalonado (`LOAD_SHAPE=steps`): +20 usuarios cada 60 s hasta 200 |
| Dónde corrió Locust | Dentro del clúster (`k8s/locust/locust.yaml`), contra `http://inference-api:8000` |
| 1 réplica | 1 CPU / 1 GB (`requests = limits`), probes por defecto (liveness con timeout de 1 s) |
| 3 réplicas | `requests 300m / limits 1 CPU`, 512Mi–1Gi, probes con timeout de 5 s |
| Criterio "soporta" | < 1 % de fallos y p95 < 500 ms |

Con `requests: 1 CPU` las 3 réplicas no cabían en la VM de 4 CPU, por eso se redujeron los `requests` (los `limits` siguen en 1 CPU por réplica).

## 2. Cómo se obtuvieron los datos

Los resultados de Locust se exportaron a `docs/resultados/`: CSV y reporte HTML por configuración (`rep1_steps_*` y `rep3_steps_*`). Cada fila de las tablas siguientes toma la **segunda mitad de cada escalón de 60 s** (ya estabilizado) de la serie temporal del reporte HTML; RPS y percentiles son los de la ventana móvil de Locust. La granularidad es de 20 usuarios, así que el límite real está entre el último nivel que cumple y el siguiente.

**Diferencias con el método del taller:** perfil escalonado en lugar de niveles fijos de 2 minutos con `--reset-stats`; Locust comparte la VM con la API, PostgreSQL, Prometheus y Airflow en vez de correr en otra máquina; y las dos configuraciones no tienen exactamente los mismos recursos ni probes. Hay que tenerlas en cuenta al interpretar las cifras: la comparación entre 1 y 3 réplicas es válida en orden de magnitud, no como medición de laboratorio.

## 3. Resultados

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

## 4. Análisis

1. **1 réplica:** la máxima concurrencia que cumple el criterio es **20 usuarios (~11 RPS)**; el límite real está entre 20 y 40 usuarios (a 40 el p95 ya es 688 ms). El techo de RPS es de ~22 y se alcanza desde ~60 usuarios.
2. **3 réplicas:** **100 usuarios (~54 RPS)**; el límite real está entre 100 y 120 (a 120 el p95 es 573 ms). El techo de RPS es de ~67 desde ~140 usuarios.
3. **Cambio de capacidad:** en usuarios sostenibles pasa de 20 a 100 (×5) y en RPS en el límite de 11 a 54 (×4,9), pero esa comparación está inflada por la granularidad de 20 usuarios (el límite de 1 réplica está entre 20 y 40). La medida más fiable es el **techo de RPS: ~22 → ~67, es decir ×3,0**, prácticamente lineal (~22 RPS por CPU). Sobre el promedio de toda la prueba el aumento es ×2,8 (17,5 → 48,4 RPS) porque la prueba de 1 réplica incluye el periodo con reinicios del pod.
4. **Recurso que se satura primero: la CPU de la réplica de la API** (un proceso uvicorn por pod, límite de 1 CPU). Evidencia: con 1 réplica el RPS se aplana en ~22 desde 60 usuarios mientras la latencia sigue creciendo, y con 3 réplicas se aplana en ~67 (×3). Pendiente de confirmar con la captura de CPU por pod en Grafana (`rate(container_cpu_usage_seconds_total)` o `kubectl top pod`), que no está en los CSV de Locust. A partir de ~120 usuarios con 1 réplica la liveness probe (timeout 1 s) reinició el pod 5 veces (fallos `Connection refused`/`reset`), lo que explica la caída de RPS a 13–18.
5. **Servicios adicionales que pudieron limitar:** los 4 CPU de la VM se comparten con Locust, PostgreSQL (que recibe un `INSERT` por predicción en `inference_logs`), Prometheus y Airflow, lo que limita cuánto más se puede escalar con réplicas en esta VM. Locust llama al Service de Kubernetes (`inference-api:8000`) sin Ingress ni Nginx intermedio. MLflow solo se consulta al arrancar y cada 60 s para ver el alias, no por petición.

## 5. Consideraciones sobre nuestro sistema

- **PostgreSQL está en la ruta de cada petición:** cada predicción se registra en `inference_logs` (en segundo plano, sin sumar latencia a la respuesta, pero con consumo de CPU y conexiones). Durante la prueba se registraron decenas de miles de filas con `source='locust'`.
- **Un proceso uvicorn por pod:** la API escala con réplicas de Kubernetes, no con workers; por eso cada réplica tiene un techo de ~22 RPS.
- **Balanceo:** el Service de Kubernetes reparte por conexión; Locust mantiene una conexión por usuario, así que con decenas de usuarios el reparto entre réplicas debe ser parejo (se puede comprobar en Grafana con `sum by (pod)`).
- **Prometheus por pod:** las métricas de cada réplica se descubren individualmente (`kubernetes_sd_configs`), lo que permite ver el reparto y la CPU por réplica en Grafana (`docs/metricas.md`).
- **Probes:** con 1 réplica saturada, `/health` tardó más de 1 s y Kubernetes reinició el pod 5 veces en plena prueba (falla en cascada); con `timeoutSeconds: 5` hubo 0 reinicios.
