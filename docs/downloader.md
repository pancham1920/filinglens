# SEC downloader checkpoint

The downloader reads `data/manifest.json`, validates filing metadata and SEC archive
URLs, and saves the original HTML in `data/cache/<accession_number>.html`.

## Setup and execution

From `backend`, install the locked dependencies with `uv sync --locked`.
Create `.env` in the repository root using `.env.example` as a template. Set a
descriptive `SEC_USER_AGENT` containing the application name and your contact
details. Do not commit `.env`.

Run from `backend`:

```bash
uv run --offline python -m app.ingestion.downloader
```

`--offline` prevents uv from fetching packages; the Python downloader still needs
network access when a filing is not cached. Dependencies must already be installed.

The first successful run downloads and atomically saves the HTML. Subsequent runs
validate and reuse the cached file without making an HTTP request for that filing.
Each manifest entry is processed; individual failures do not stop later filings.

The final `SUMMARY` reports `downloaded`, `cached`, and `failed` counts. Exit status
is 0 when every filing was downloaded or cached and 1 on configuration errors or
any filing failure. Malformed manifests stop before processing entries.

## Automated verification

```bash
uv run --offline python -m unittest discover -s tests -v
```

Tests use temporary directories and HTTPX MockTransport, not the SEC or model APIs.
They cover metadata/URL validation, HTML checks, atomic writes and cleanup, cache
reuse, bounded retries, Retry-After, batch continuation, summaries, and exit status.

## Access and limitations

- Uses your configured User-Agent and keeps at least one second between requests
  within this process. Other simultaneous processes have independent limits.
- Tries a request at most three times. Honors Retry-After when retrying, or stops
  if the requested delay exceeds 60 seconds. Permanent HTTP errors are not retried.
- Never disables TLS verification or bypasses organization network restrictions.
- HTML validation is a sanity check and rejects known SEC access-limit pages; it
  does not prove document identity, completeness, or factual accuracy.
- Atomic replacement prevents ordinary partial writes from appearing at the final
  cache path. It is not a guarantee of durability after sudden power loss.
- No parsing, chunking, embeddings, or database ingestion is implemented here.
- Docker and PostgreSQL/pgvector setup remain separate, deferred work.

SEC access guidance: https://www.sec.gov/about/developer-resources

## Checkpoint verification

- All 21 saved automated tests passed.
- The existing Apple filing cache (1,503,889 bytes) passed the HTML checks. A normal
  application run reported downloaded=0, cached=1, failed=0 and exit status 0.
  HTTP client creation was forbidden during this check, proving network was skipped.
- A fresh download into a separate empty verification cache was attempted, but
  all three connection attempts failed. It reported downloaded=0, cached=0,
  failed=1 and exit status 1, and no invalid content was saved.
- A fresh live download must be reverified when SEC connectivity is available.
  Automated download/retry behavior is covered by mocked responses, not a claim
  that a live download succeeded during this checkpoint.
