-- Only llm_ tables belong to this service.

CREATE TABLE IF NOT EXISTS llm_models (
    id VARCHAR(64) PRIMARY KEY,
    upstream_model_id LONGTEXT NOT NULL,
    name LONGTEXT NOT NULL,
    description LONGTEXT,
    base_url LONGTEXT NOT NULL,
    encrypted_api_key LONGTEXT NOT NULL,
    created_at VARCHAR(40) NOT NULL,
    updated_at VARCHAR(40) NOT NULL,
    deleted_at VARCHAR(40)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS llm_call_records (
    id VARCHAR(64) PRIMARY KEY,
    model_id VARCHAR(64) NOT NULL,
    token_hash VARCHAR(64) NOT NULL,
    input LONGTEXT NOT NULL,
    output_json LONGTEXT,
    response_type LONGTEXT NOT NULL,
    status VARCHAR(32) NOT NULL CHECK (status IN ('success', 'error')),
    error_code LONGTEXT,
    created_at VARCHAR(40) NOT NULL,
    FOREIGN KEY (model_id) REFERENCES llm_models(id),
    INDEX ix_call_records_token_created_at (token_hash, created_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS llm_analysis_records (
    id VARCHAR(64) PRIMARY KEY,
    token_hash VARCHAR(64) NOT NULL,
    kind VARCHAR(64) NOT NULL,
    scenario_code LONGTEXT,
    model_id VARCHAR(64) NOT NULL,
    input_json LONGTEXT NOT NULL,
    state_json LONGTEXT NOT NULL,
    status VARCHAR(32) NOT NULL CHECK(status IN ('running', 'succeeded', 'failed')),
    created_at VARCHAR(40) NOT NULL,
    FOREIGN KEY (model_id) REFERENCES llm_models(id),
    INDEX ix_analysis_records_token_created (token_hash, created_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS llm_neural_trace_tasks (
    id VARCHAR(64) PRIMARY KEY,
    token_hash VARCHAR(64) NOT NULL,
    case_id VARCHAR(255) NOT NULL,
    input_json LONGTEXT NOT NULL,
    status VARCHAR(32) NOT NULL CHECK(status IN ('pending', 'running', 'succeeded', 'failed')),
    stage LONGTEXT NOT NULL,
    result_json LONGTEXT,
    error_message LONGTEXT,
    created_at VARCHAR(40) NOT NULL,
    updated_at VARCHAR(40) NOT NULL,
    started_at VARCHAR(40),
    finished_at VARCHAR(40),
    INDEX ix_neural_trace_tasks_token_created (token_hash, created_at DESC),
    INDEX ix_pending (status, created_at, id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS llm_governance_tasks (
    id VARCHAR(64) PRIMARY KEY,
    kind VARCHAR(64) NOT NULL,
    token_hash VARCHAR(64) NOT NULL,
    name LONGTEXT NOT NULL,
    scope_json LONGTEXT NOT NULL,
    model_id VARCHAR(64) NOT NULL,
    model_version LONGTEXT NOT NULL,
    status VARCHAR(32) NOT NULL CHECK (status IN ('pending', 'running', 'succeeded', 'failed')),
    result_id LONGTEXT,
    error_message LONGTEXT,
    created_at VARCHAR(40) NOT NULL,
    updated_at VARCHAR(40) NOT NULL,
    INDEX ix_governance_tasks_scope (token_hash, kind, scope_json(191), created_at DESC),
    INDEX ix_pending (status, created_at, id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS llm_governance_task_samples (
    task_id VARCHAR(64) NOT NULL,
    sample_id VARCHAR(255) NOT NULL,
    status VARCHAR(32) NOT NULL CHECK (status IN ('success', 'error')),
    output_json LONGTEXT,
    error_code LONGTEXT,
    PRIMARY KEY (task_id, sample_id),
    FOREIGN KEY (task_id) REFERENCES llm_governance_tasks(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS llm_governance_results (
    id VARCHAR(64) PRIMARY KEY,
    task_id VARCHAR(64) NOT NULL UNIQUE,
    kind VARCHAR(64) NOT NULL,
    token_hash VARCHAR(64) NOT NULL,
    scope_json LONGTEXT NOT NULL,
    summary_json LONGTEXT NOT NULL,
    samples_json LONGTEXT NOT NULL,
    created_at VARCHAR(40) NOT NULL,
    FOREIGN KEY (task_id) REFERENCES llm_governance_tasks(id),
    INDEX ix_governance_results_scope (token_hash, kind, scope_json(191), created_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS llm_auth_tokens (
    id VARCHAR(64) PRIMARY KEY,
    token_hash VARCHAR(64) NOT NULL UNIQUE,
    label LONGTEXT NOT NULL DEFAULT (''),
    created_at VARCHAR(40) NOT NULL,
    expires_at VARCHAR(40) NOT NULL,
    enabled BIGINT NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    deleted_at VARCHAR(40)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS llm_governance_model_bindings (
    kind VARCHAR(64) PRIMARY KEY,
    model_id VARCHAR(64) NOT NULL,
    updated_at VARCHAR(40) NOT NULL,
    FOREIGN KEY (model_id) REFERENCES llm_models(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS llm_business_database_config (
    id BIGINT PRIMARY KEY CHECK (id = 1),
    host LONGTEXT NOT NULL,
    port BIGINT NOT NULL,
    database_name LONGTEXT NOT NULL,
    username LONGTEXT NOT NULL,
    encrypted_password LONGTEXT NOT NULL,
    updated_at VARCHAR(40) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS llm_training_tasks (
    id VARCHAR(64) PRIMARY KEY,
    token_hash VARCHAR(64) NOT NULL,
    config_json LONGTEXT NOT NULL,
    source_hash VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL CHECK(status IN
    ('pending', 'running', 'canceling', 'canceled', 'succeeded', 'failed')),
    epoch BIGINT NOT NULL DEFAULT 0,
    step BIGINT NOT NULL DEFAULT 0,
    progress DOUBLE NOT NULL DEFAULT 0,
    latest_metrics_json LONGTEXT NOT NULL DEFAULT ('{}'),
    runtime_json LONGTEXT NOT NULL DEFAULT ('{}'),
    error_message LONGTEXT,
    created_at VARCHAR(40) NOT NULL,
    updated_at VARCHAR(40) NOT NULL,
    started_at VARCHAR(40),
    finished_at VARCHAR(40),
    INDEX ix_training_tasks_token_created (token_hash, created_at DESC),
    INDEX ix_pending (status, created_at, id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS llm_training_metrics (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    task_id VARCHAR(64) NOT NULL,
    epoch BIGINT NOT NULL,
    step BIGINT NOT NULL,
    phase VARCHAR(32) NOT NULL CHECK(phase IN ('train', 'validation')),
    metrics_json LONGTEXT NOT NULL,
    created_at VARCHAR(40) NOT NULL,
    FOREIGN KEY (task_id) REFERENCES llm_training_tasks(id),
    INDEX ix_training_metrics_task_id (task_id, id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS llm_training_script_events (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    task_id VARCHAR(64) NOT NULL,
    type VARCHAR(32) NOT NULL CHECK(type IN ('metric', 'progress', 'artifact')),
    event_json LONGTEXT NOT NULL,
    created_at VARCHAR(40) NOT NULL,
    FOREIGN KEY (task_id) REFERENCES llm_training_tasks(id),
    INDEX ix_training_script_events_task_id (task_id, id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS llm_training_artifacts (
    id VARCHAR(64) PRIMARY KEY,
    task_id VARCHAR(64) NOT NULL,
    filename VARCHAR(512) NOT NULL,
    kind VARCHAR(64) NOT NULL,
    epoch BIGINT,
    loss DOUBLE,
    size BIGINT NOT NULL,
    sha256 VARCHAR(64) NOT NULL,
    UNIQUE(task_id, filename),
    FOREIGN KEY (task_id) REFERENCES llm_training_tasks(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
