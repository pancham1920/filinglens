BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

-- Companies whose filings we index.
CREATE TABLE companies (
    cik TEXT PRIMARY KEY
        CHECK (cik ~ '^[0-9]{10}$'),
    ticker TEXT NOT NULL,
    name TEXT NOT NULL
);

-- One record per SEC filing.
CREATE TABLE filings (
    accession_number TEXT PRIMARY KEY
        CHECK (accession_number ~ '^[0-9]{10}-[0-9]{2}-[0-9]{6}$'),
    company_cik TEXT NOT NULL REFERENCES companies(cik),
    form_type TEXT NOT NULL CHECK (form_type = '10-K'),
    fiscal_year INTEGER NOT NULL,
    period_end DATE NOT NULL,
    filed_at DATE NOT NULL,
    source_url TEXT NOT NULL
);

-- Tracks which processing version produced a filing's chunks.
CREATE TABLE ingestion_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filing_accession TEXT NOT NULL
        REFERENCES filings(accession_number),
    pipeline_version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'processing'
        CHECK (status IN ('processing', 'completed', 'failed')),
    active BOOLEAN NOT NULL DEFAULT FALSE,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,

    -- Only completed runs may become searchable.
    CHECK (NOT active OR status = 'completed')
);

-- At most one processing version is active for each filing.
CREATE UNIQUE INDEX one_active_run_per_filing
    ON ingestion_runs(filing_accession)
    WHERE active = TRUE;

-- Searchable passages and their embeddings.
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ingestion_run_id UUID NOT NULL REFERENCES ingestion_runs(id),
    section TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    content_type TEXT NOT NULL
        CHECK (content_type IN ('prose', 'table')),
    content TEXT NOT NULL,
    source_locator TEXT,
    embedding VECTOR,

    -- Repeated inserts cannot duplicate the same chunk position.
    UNIQUE (ingestion_run_id, position)
);

CREATE INDEX filings_company_year
    ON filings(company_cik, fiscal_year);

CREATE INDEX ingestion_runs_filing
    ON ingestion_runs(filing_accession);

COMMIT;