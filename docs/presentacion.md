# Guion de presentación y plan del video

Ajustar tiempos según el límite que fije el enunciado/profe. Propuesta: **~10 min**, 2 voces (Ana: plataforma · Daniel: datos, modelo y aplicación).

## Hilo conductor (una frase)
"Un pipeline automático que carga datos por lotes, entrena y promueve un modelo, lo sirve por una API con métricas, y probamos cuánta carga aguanta."

## Guion por bloques

| # | Min | Quién | Qué decir / mostrar |
|---|---|---|---|
| 1 | 0:45 | Ambos | Problema (readmisión < 30 días en pacientes con diabetes) y por qué importa. Mostrar el diagrama de arquitectura. |
| 2 | 1:15 | Ana | Plataforma en Kubernetes: Airflow, PostgreSQL, MLflow + MinIO, API, Streamlit, Prometheus/Grafana, imágenes en DockerHub. Cómo se despliega. |
| 3 | 1:30 | Daniel | **Datos**: `raw_data` → `clean_data`; lotes de 15.000, idempotencia, partición por paciente (evitar fuga). Mostrar `ingestion_batches` y conteos. |
| 4 | 1:15 | Daniel | **DAG en Airflow**: mostrar el grafo y una ejecución en vivo/grabada; lote 2; archivo agotado = *skipped*. |
| 5 | 1:30 | Daniel | **Experimentación**: 3 modelos en MLflow, métrica F2 y por qué, umbral optimizado, resultados (tabla), comparación y promoción a `champion`. Aclarar el compromiso recall alto / precisión baja. |
| 6 | 1:15 | Daniel | **Inferencia**: API (carga por alias, refresco sin reiniciar, validación 422, logging a `inference_logs`) y Streamlit (demo: predicción válida + error). |
| 7 | 1:00 | Ana | **Observabilidad**: tablero de Grafana con las métricas del enunciado. |
| 8 | 1:30 | Ambos | **Pruebas de carga**: escenarios A/B/C, tabla de resultados, **punto de degradación**, recomendación de réplicas. |
| 9 | 0:45 | Ambos | **Dificultades y aprendizajes**, y qué mejoraríamos (datos reales continuos, monitoreo de deriva, CI/CD). |

## Demo en vivo recomendada (orden)
1. Airflow → lanzar el DAG (o mostrar la ejecución terminada) → MLflow: nuevas corridas y registro.
2. Si se promueve un champion nuevo: `/model-info` cambia de versión **sin reiniciar** (evidencia 20).
3. Streamlit: predicción con el ejemplo 2 (riesgo alto) y luego un JSON inválido.
4. Grafana moviéndose mientras Locust corre.

## Preguntas probables (preparar respuesta corta)
- ¿Por qué F2 y no accuracy? → desbalance (~11 % positivos) y el costo de no detectar un paciente de riesgo.
- ¿Por qué el umbral no es 0,5? → se optimiza en validación para F2 y viaja con el modelo.
- ¿Cómo evitan fuga de datos? → partición por paciente con hash determinista.
- ¿Qué pasa si MLflow se cae? → la API sigue con el último modelo bueno; `/predict` nunca llama a MLflow.
- ¿Qué pasa si el DAG se ejecuta dos veces / falla a la mitad? → idempotente; se retoma el lote incompleto.
- ¿Cuándo se degrada la API y qué hicieron al respecto? → resultados de `pruebas_de_carga.md`.
- ¿Por qué un proceso por pod? → el escalado lo gestiona Kubernetes con réplicas; así CPU/límites son predecibles.

## Plan del video
- **Formato**: grabación de pantalla con voz (Zoom/Meet/OBS), subir a YouTube (*no listado*), y poner el enlace en el README. Verificar el límite de duración y entregables del enunciado.
- **Antes de grabar**: todo desplegado y estable; DAG ya ejecutado al menos 2 veces; pestañas abiertas en orden (Airflow, MLflow, Streamlit, Grafana, Locust, terminal con `kubectl get pods`); tamaño de letra grande; notificaciones apagadas.
- **Grabación**: ensayar una vez completo; grabar por bloques (la tabla de arriba) y unir si hace falta; si algo tarda (entrenamiento), mostrar la ejecución ya terminada en vez de esperar.
- **Cierre**: revisar audio, que se lean los textos en pantalla y que no aparezcan contraseñas/secretos.
