# Local embeddings: section 2.5

FilingLens uses `BAAI/bge-small-en-v1.5` through Sentence Transformers on the
CPU. It generates 384-dimensional, normalized vectors without calling a paid
embedding API. No OpenAI API key is used by this module.

## Setup

Run from `backend`:

```bash
uv sync
uv run --offline python -m app.db.migrate_local_embeddings
uv run --offline python -m app.ingestion.embeddings
```

The migration requires migrations 001 and 002 to have already been applied.
It refuses to resize existing embeddings or ingestion runs with incompatible
dimensions. Migration 002 is historical; migration 003 switches its schema to
384 dimensions. Do not reapply 002 after 003.

`uv sync` needs access to PyPI on the first installation. The first embedding
run also downloads the public model from Hugging Face into `data/models/`,
which is ignored by Git. `uv run --offline` only disables uv's dependency
downloads; it does not disable model downloads inside the Python program.

Once the model has been downloaded, test genuinely offline model loading with:

```bash
uv run --offline python -m app.ingestion.embeddings --offline
```

Successful output includes `Dimensions: 384` and
`PASS: generated a valid embedding locally (no API call)`.

## Prevent silent truncation

The model accepts at most 512 tokens including special tokens. Word counts do
not establish token counts. The embedding function counts with the model's
tokenizer and rejects oversized input before calling `encode`.

Audit the saved 2.4 export:

```bash
uv run --offline python -m app.ingestion.embeddings --offline --check-chunks
```

An audit exits with status 1 when splitting is required, and lists the affected
sections and positions. It never modifies the export. Token-aware splitting,
batching, vector caching, and database ingestion belong to the following 2.5
steps; the current CLI tests one short sentence.

Passages use their original text. Future search queries use
`embed_text(question, is_query=True)` to add BGE's retrieval instruction.
Document and query vectors must come from this same model and preprocessing.

## Tests

```bash
uv run --offline python -m unittest discover -s tests -v
```

The automated embedding tests use a fake model and tokenizer. They check
boundary lengths, query instructions, rejection before truncation, vector
validation, and non-mutating audits; they do not establish model download or
real inference success. Verify those separately using the CLI above.

Model documentation: https://huggingface.co/BAAI/bge-small-en-v1.5
