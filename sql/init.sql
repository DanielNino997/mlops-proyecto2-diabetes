-- =====================================================================
-- Proyecto 2 MLOps: esquema de la base de datos del proyecto
-- (la base de metadatos de MLflow es aparte y la crea MLflow).
--
-- Tres capas separadas:
--   raw_data        -> datos tal como llegan del archivo (carril A)
--   clean_data      -> datos limpios para entrenar (carril A, columnas de B)
--   inference_logs  -> cada predicción que hace la API (carril B)
--
-- Todo usa IF NOT EXISTS para que el script se pueda correr varias veces.
-- =====================================================================


-- ---------------------------------------------------------------------
-- raw_data y clean_data: PENDIENTE (carril A)
-- A completa estas dos tablas. clean_data debe tener:
--   - las columnas de src/schemas.py (FEATURE_COLUMNS),
--   - patient_nbr (para el split por paciente),
--   - readmitted_lt30 SMALLINT (target 0/1),
--   - batch_id y row_hash (para trazar de qué lote vino cada fila).
-- ---------------------------------------------------------------------


-- ---------------------------------------------------------------------
-- inference_logs: una fila por cada llamada a /predict
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inference_logs (
    id              BIGSERIAL PRIMARY KEY,

    -- Identificador de la solicitud; la API lo devuelve en la respuesta.
    request_id      UUID NOT NULL UNIQUE,

    -- Fecha y hora de la inferencia.
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Datos de entrada tal como llegaron a la API. JSONB evita tener que
    -- cambiar esta tabla si cambian las features.
    input_data      JSONB NOT NULL,

    -- Resultado. En una solicitud fallida prediction y probability quedan
    -- vacías y error_message explica qué pasó.
    status          TEXT NOT NULL DEFAULT 'success'
                    CHECK (status IN ('success', 'error')),
    prediction      SMALLINT CHECK (prediction IN (0, 1)),
    probability     DOUBLE PRECISION CHECK (probability BETWEEN 0 AND 1),
    threshold       DOUBLE PRECISION CHECK (threshold BETWEEN 0 AND 1),
    error_message   TEXT,

    -- Qué modelo respondió.
    model_name      TEXT NOT NULL,
    model_version   TEXT,
    model_alias     TEXT NOT NULL,

    -- Tiempo de respuesta de la inferencia, en milisegundos.
    latency_ms      DOUBLE PRECISION NOT NULL CHECK (latency_ms >= 0),

    -- Quién llamó (encabezado X-Client: 'streamlit', 'locust', ...).
    -- Sirve para no mezclar el tráfico de las pruebas de carga con
    -- inferencias reales si después se reentrena con esta tabla.
    source          TEXT NOT NULL DEFAULT 'unknown',

    -- Una solicitud exitosa siempre tiene predicción y probabilidad.
    CONSTRAINT success_has_prediction CHECK (
        status <> 'success'
        OR (prediction IS NOT NULL AND probability IS NOT NULL)
    )
);

-- Consultas típicas: "inferencias de la última hora" y "por versión".
CREATE INDEX IF NOT EXISTS idx_inference_logs_created_at
    ON inference_logs (created_at);

CREATE INDEX IF NOT EXISTS idx_inference_logs_model_version
    ON inference_logs (model_name, model_version);
