-- =====================================================================
-- Complemento del pipeline (carril B). Se ejecuta DESPUÉS de init.sql, desde la
-- primera tarea del DAG (db.init_schema). Solo agrega lo que init.sql no trae;
-- usa IF NOT EXISTS, así que se puede correr varias veces.
--
--   ingestion_batches  qué filas del archivo forman cada lote (carga incremental)
--   clean_data.split   conjunto (train/val/test) asignado a cada paciente
-- =====================================================================

-- dag_run_id UNIQUE: reintentar la tarea de carga dentro de la misma ejecución
-- reutiliza el mismo lote en vez de avanzar al siguiente.
CREATE TABLE IF NOT EXISTS ingestion_batches (
    batch_id        INTEGER PRIMARY KEY,
    dag_run_id      TEXT UNIQUE,
    source_file     TEXT NOT NULL,
    row_start       INTEGER NOT NULL,          -- primera fila del archivo (desde 0)
    row_end         INTEGER NOT NULL,          -- fila final, no incluida
    status          TEXT NOT NULL DEFAULT 'planned'
                    CHECK (status IN ('planned', 'loaded', 'validated', 'processed', 'failed')),
    loaded_rows     INTEGER,
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE clean_data ADD COLUMN IF NOT EXISTS split TEXT
    CHECK (split IN ('train', 'val', 'test'));
CREATE INDEX IF NOT EXISTS idx_clean_split ON clean_data (split);
