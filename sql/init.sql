-- Proyecto 2 MLOps - esquema de la base mlops_data
-- Capas: raw_data (crudo), clean_data (limpio), inference_logs (API)

-- RAW
CREATE TABLE IF NOT EXISTS raw_data (
    id           BIGSERIAL    PRIMARY KEY,
    batch_id     INTEGER      NOT NULL,
    load_ts      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    source_file  TEXT         NOT NULL,
    row_hash     CHAR(64)     NOT NULL UNIQUE,
    status       TEXT         NOT NULL DEFAULT 'loaded',
    payload      JSONB        NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_raw_batch  ON raw_data (batch_id);
CREATE INDEX IF NOT EXISTS idx_raw_status ON raw_data (status);

-- CLEAN
CREATE TABLE IF NOT EXISTS clean_data (
    id                        BIGSERIAL   PRIMARY KEY,
    encounter_id              BIGINT      NOT NULL UNIQUE,
    patient_nbr               BIGINT      NOT NULL,
    race                      TEXT        NOT NULL,
    gender                    TEXT        NOT NULL,
    age                       TEXT        NOT NULL,
    admission_type_id         INTEGER     NOT NULL,
    discharge_disposition_id  INTEGER     NOT NULL,
    admission_source_id       INTEGER     NOT NULL,
    time_in_hospital          INTEGER     NOT NULL,
    medical_specialty         TEXT        NOT NULL,
    num_lab_procedures        INTEGER     NOT NULL,
    num_procedures            INTEGER     NOT NULL,
    num_medications           INTEGER     NOT NULL,
    number_diagnoses          INTEGER     NOT NULL,
    number_outpatient         INTEGER     NOT NULL,
    number_emergency          INTEGER     NOT NULL,
    number_inpatient          INTEGER     NOT NULL,
    diag_1                    TEXT        NOT NULL,
    diag_2                    TEXT        NOT NULL,
    diag_3                    TEXT        NOT NULL,
    max_glu_serum             TEXT        NOT NULL,
    "A1Cresult"               TEXT        NOT NULL,
    insulin                   TEXT        NOT NULL,
    metformin                 TEXT        NOT NULL,
    change                    TEXT        NOT NULL,
    "diabetesMed"             TEXT        NOT NULL,
    readmitted_lt30           SMALLINT    NOT NULL CHECK (readmitted_lt30 IN (0, 1)),
    batch_id                  INTEGER     NOT NULL,
    row_hash                  CHAR(64)    NOT NULL REFERENCES raw_data(row_hash),
    processed_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_clean_patient ON clean_data (patient_nbr);

-- INFERENCE LOGS
CREATE TABLE IF NOT EXISTS inference_logs (
    id              BIGSERIAL PRIMARY KEY,
    request_id      UUID NOT NULL UNIQUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    input_data      JSONB NOT NULL,
    status          TEXT NOT NULL DEFAULT 'success'
                    CHECK (status IN ('success', 'error')),
    prediction      SMALLINT CHECK (prediction IN (0, 1)),
    probability     DOUBLE PRECISION CHECK (probability BETWEEN 0 AND 1),
    threshold       DOUBLE PRECISION CHECK (threshold BETWEEN 0 AND 1),
    error_message   TEXT,
    model_name      TEXT NOT NULL,
    model_version   TEXT,
    model_alias     TEXT NOT NULL,
    latency_ms      DOUBLE PRECISION NOT NULL CHECK (latency_ms >= 0),
    source          TEXT NOT NULL DEFAULT 'unknown',
    CONSTRAINT success_has_prediction CHECK (
        status <> 'success'
        OR (prediction IS NOT NULL AND probability IS NOT NULL)
    )
);
CREATE INDEX IF NOT EXISTS idx_inference_logs_created_at
    ON inference_logs (created_at);
CREATE INDEX IF NOT EXISTS idx_inference_logs_model_version
    ON inference_logs (model_name, model_version);