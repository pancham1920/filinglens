-- Follow-up to the already-applied 002 migration. The caller owns the transaction.
LOCK TABLE chunks, ingestion_runs IN ACCESS EXCLUSIVE MODE;

-- Different models' vectors cannot be converted just by changing dimensions.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM chunks
        WHERE embedding IS NOT NULL AND vector_dims(embedding) <> 384
    ) OR EXISTS (
        SELECT 1 FROM ingestion_runs
        WHERE embedding_dimensions IS NOT NULL AND embedding_dimensions <> 384
    ) THEN
        RAISE EXCEPTION 'Existing embeddings use another model dimension; re-ingestion is required';
    END IF;
END $$;

ALTER TABLE chunks
    ALTER COLUMN embedding TYPE vector(384);

ALTER TABLE ingestion_runs
    DROP CONSTRAINT IF EXISTS ingestion_runs_embedding_dimensions_check;
ALTER TABLE ingestion_runs
    ADD CONSTRAINT ingestion_runs_embedding_dimensions_check
    CHECK (embedding_dimensions = 384);
