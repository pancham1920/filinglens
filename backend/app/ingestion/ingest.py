"""Store prepared chunk/vector pairs in PostgreSQL."""

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urldefrag
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb
from pydantic import ValidationError

from app.db.connection import PROJECT_ROOT, get_connection
from app.ingestion.chunker import Chunk, generate_chunk_id
from app.ingestion.downloader import Filing, Manifest
from app.ingestion.embedding_cache import (
    embedding_profile,
    make_cache_key,
)
from app.ingestion.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EmbeddingError,
    validate_vector,
)


ACCESSION = "0000320193-24-000123"
PIPELINE_VERSION = "ingestion-v1"

MANIFEST_PATH = PROJECT_ROOT / "data/manifest.json"

INPUT_PATH = (
    PROJECT_ROOT
    / "data/embedded_chunks"
    / f"{ACCESSION}.jsonl"
)

PREPARED_PATH = (
    PROJECT_ROOT
    / "data/prepared_chunks"
    / f"{ACCESSION}.jsonl"
)


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: Chunk
    vector: list[float]
    cache_key: str


def load_filing() -> Filing:
    manifest = Manifest.model_validate_json(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    for filing in manifest.filings:
        if filing.accession_number == ACCESSION:
            return filing

    raise ValueError("Filing is missing from manifest")


def load_records(
    path: Path,
    filing: Filing,
) -> list[EmbeddedChunk]:
    records = []
    profile = embedding_profile()

    lines = path.read_text(encoding="utf-8").splitlines()

    for number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"Blank export line: {number}")

        raw = json.loads(line)

        if not isinstance(raw, dict) or set(raw) != {
            "chunk",
            "embedding_profile",
            "cache_key",
            "embedding",
        }:
            raise ValueError(
                f"Unexpected record structure at line {number}"
            )

        if raw["embedding_profile"] != profile:
            raise ValueError(
                f"Embedding profile mismatch at line {number}"
            )

        chunk = Chunk.model_validate(raw["chunk"])
        vector = validate_vector(raw["embedding"])

        if chunk.oversized:
            raise ValueError("An oversized chunk reached ingestion")

        if (
            chunk.accession_number != filing.accession_number
            or chunk.company != filing.company
            or chunk.ticker != filing.ticker
            or chunk.fiscal_year != filing.fiscal_year
        ):
            raise ValueError(
                f"Filing metadata mismatch at line {number}"
            )

        citation_base, _ = urldefrag(chunk.citation_url)

        if citation_base != filing.source_url:
            raise ValueError(
                f"Citation does not match filing at line {number}"
            )

        expected_id = generate_chunk_id(
            accession_number=chunk.accession_number,
            section=chunk.section,
            position=chunk.position,
            content_type=chunk.content_type,
            text=chunk.text,
            chunking_version=chunk.chunking_version,
        )

        if chunk.chunk_id != expected_id:
            raise ValueError(
                f"Chunk ID mismatch at line {number}"
            )

        # Also confirm it can be stored in a UUID column.
        UUID(chunk.chunk_id)

        if raw["cache_key"] != make_cache_key(chunk.text):
            raise ValueError(
                f"Cache identity mismatch at line {number}"
            )

        # The selected embedding profile uses normalized vectors.
        squared_norm = sum(value * value for value in vector)

        if not math.isclose(
            squared_norm,
            1.0,
            rel_tol=0.001,
            abs_tol=0.001,
        ):
            raise ValueError(
                f"Vector is not normalized at line {number}"
            )

        records.append(
            EmbeddedChunk(
                chunk=chunk,
                vector=vector,
                cache_key=raw["cache_key"],
            )
        )

    if not records:
        raise ValueError("Embedded export is empty")

    ids = [record.chunk.chunk_id for record in records]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate exported chunk IDs")

    locations = [
        (record.chunk.section, record.chunk.position)
        for record in records
    ]

    if len(locations) != len(set(locations)):
        raise ValueError("Duplicate section positions")

    # Detect missing or changed chunk records.
    prepared = [
        Chunk.model_validate_json(line)
        for line in PREPARED_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    if [record.chunk for record in records] != prepared:
        raise ValueError(
            "Embedded records do not match the prepared export"
        )

    return records


def content_hash(
    filing: Filing,
    records: list[EmbeddedChunk],
) -> str:
    """Fingerprint metadata, chunks, vectors, and processing settings."""

    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "embedding_profile": embedding_profile(),
        "filing": filing.model_dump(mode="json"),
        "records": [
            {
                "chunk": record.chunk.model_dump(mode="json"),
                "cache_key": record.cache_key,
                "embedding": record.vector,
            }
            for record in records
        ],
    }

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def source_metadata(record: EmbeddedChunk) -> dict:
    # Text is stored separately in chunks.content.
    metadata = record.chunk.model_dump(
        mode="json",
        exclude={"text"},
    )

    metadata["embedding_profile"] = embedding_profile()
    metadata["cache_key"] = record.cache_key

    return metadata


def ensure_filing(
    connection: psycopg.Connection,
    filing: Filing,
) -> None:
    """Create missing metadata without overwriting existing records."""

    connection.execute(
        """
        INSERT INTO companies (cik, ticker, name)
        VALUES (%s, %s, %s)
        ON CONFLICT (cik) DO NOTHING
        """,
        (filing.cik, filing.ticker, filing.company),
    )

    company = connection.execute(
        """
        SELECT ticker, name
        FROM companies
        WHERE cik = %s
        """,
        (filing.cik,),
    ).fetchone()

    if company != (filing.ticker, filing.company):
        raise ValueError(
            "Existing company metadata differs from the manifest"
        )

    connection.execute(
        """
        INSERT INTO filings (
            accession_number,
            company_cik,
            form_type,
            fiscal_year,
            period_end,
            filed_at,
            source_url
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (accession_number) DO NOTHING
        """,
        (
            filing.accession_number,
            filing.cik,
            filing.form_type,
            filing.fiscal_year,
            filing.period_end,
            filing.filed_at,
            filing.source_url,
        ),
    )

    stored = connection.execute(
        """
        SELECT
            company_cik,
            form_type,
            fiscal_year,
            period_end,
            filed_at,
            source_url
        FROM filings
        WHERE accession_number = %s
        """,
        (filing.accession_number,),
    ).fetchone()

    expected = (
        filing.cik,
        filing.form_type,
        filing.fiscal_year,
        filing.period_end,
        filing.filed_at,
        filing.source_url,
    )

    if stored != expected:
        raise ValueError(
            "Existing filing metadata differs from the manifest"
        )


def verify_run(
    connection: psycopg.Connection,
    run_id: UUID,
    records: list[EmbeddedChunk],
) -> None:
    """Verify stored identities, text, citations, metadata, and dimensions."""

    rows = connection.execute(
        """
        SELECT
            chunk_id,
            section,
            position,
            content_type,
            content,
            source_locator,
            source_metadata,
            vector_dims(embedding)
        FROM chunks
        WHERE ingestion_run_id = %s
        """,
        (run_id,),
    ).fetchall()

    if len(rows) != len(records):
        raise ValueError("Stored chunk count does not match input")

    by_id = {str(row[0]): row for row in rows}

    if len(by_id) != len(records):
        raise ValueError("Stored chunk identities are invalid")

    for record in records:
        chunk = record.chunk
        row = by_id.get(chunk.chunk_id)

        if row is None:
            raise ValueError("A stored chunk is missing")

        expected = (
            chunk.section,
            chunk.position,
            chunk.content_type,
            chunk.text,
            chunk.citation_url,
            source_metadata(record),
            EMBEDDING_DIMENSIONS,
        )

        if row[1:] != expected:
            raise ValueError(
                "Stored text, metadata, citation, or dimensions differ"
            )


def store_records(
    filing: Filing,
    records: list[EmbeddedChunk],
    fingerprint: str,
) -> tuple[str, UUID]:
    with get_connection() as connection:
        # Explicit transaction blocks below own commit/rollback.
        connection.autocommit = True

        locked = connection.execute(
            """
            SELECT pg_try_advisory_lock(
                hashtextextended(%s, 0)
            )
            """,
            (f"filinglens:{filing.accession_number}",),
        ).fetchone()

        if locked != (True,):
            raise RuntimeError(
                "This filing is already being ingested"
            )

        # The session lock is released when this connection closes.
        column = connection.execute(
            """
            SELECT format_type(atttypid, atttypmod)
            FROM pg_attribute
            WHERE attrelid = 'public.chunks'::regclass
              AND attname = 'embedding'
              AND NOT attisdropped
            """
        ).fetchone()

        if column != ("vector(384)",):
            raise ValueError(
                "Database schema must use vector(384)"
            )

        existing = connection.execute(
            """
            SELECT id
            FROM ingestion_runs
            WHERE filing_accession = %s
              AND pipeline_version = %s
              AND content_hash = %s
              AND embedding_model = %s
              AND embedding_dimensions = %s
              AND status = 'completed'
              AND active = TRUE
            """,
            (
                filing.accession_number,
                PIPELINE_VERSION,
                fingerprint,
                EMBEDDING_MODEL,
                EMBEDDING_DIMENSIONS,
            ),
        ).fetchone()

        if existing is not None:
            # Verify before treating the previous result as reusable.
            ensure_expected = connection.execute(
                """
                SELECT company_cik, fiscal_year, source_url
                FROM filings
                WHERE accession_number = %s
                """,
                (filing.accession_number,),
            ).fetchone()

            if ensure_expected != (
                filing.cik,
                filing.fiscal_year,
                filing.source_url,
            ):
                raise ValueError("Stored filing metadata differs")

            verify_run(connection, existing[0], records)
            return "skipped", existing[0]

        # Persist the processing run so ordinary failures can be recorded.
        with connection.transaction():
            ensure_filing(connection, filing)

            row = connection.execute(
                """
                INSERT INTO ingestion_runs (
                    filing_accession,
                    pipeline_version,
                    content_hash,
                    status,
                    active,
                    embedding_model,
                    embedding_dimensions
                )
                VALUES (%s, %s, %s, 'processing', FALSE, %s, %s)
                RETURNING id
                """,
                (
                    filing.accession_number,
                    PIPELINE_VERSION,
                    fingerprint,
                    EMBEDDING_MODEL,
                    EMBEDDING_DIMENSIONS,
                ),
            ).fetchone()

            if row is None:
                raise RuntimeError("Could not create ingestion run")

            run_id = row[0]

        try:
            # Chunks and active-run replacement commit together.
            with connection.transaction():
                parameters = []

                for record in records:
                    chunk = record.chunk

                    parameters.append((
                        run_id,
                        UUID(chunk.chunk_id),
                        chunk.section,
                        chunk.position,
                        chunk.content_type,
                        chunk.text,
                        chunk.citation_url,
                        Jsonb(source_metadata(record)),
                        json.dumps(
                            record.vector,
                            allow_nan=False,
                            separators=(",", ":"),
                        ),
                    ))

                with connection.cursor() as cursor:
                    cursor.executemany(
                        """
                        INSERT INTO chunks (
                            ingestion_run_id,
                            chunk_id,
                            section,
                            position,
                            content_type,
                            content,
                            source_locator,
                            source_metadata,
                            embedding
                        )
                        VALUES (
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s::vector(384)
                        )
                        """,
                        parameters,
                    )

                verify_run(connection, run_id, records)

                connection.execute(
                    """
                    UPDATE ingestion_runs
                    SET active = FALSE
                    WHERE filing_accession = %s
                      AND active = TRUE
                    """,
                    (filing.accession_number,),
                )

                connection.execute(
                    """
                    UPDATE ingestion_runs
                    SET status = 'completed',
                        active = TRUE,
                        completed_at = now(),
                        error_message = NULL
                    WHERE id = %s
                    """,
                    (run_id,),
                )

        except Exception as error:
            # The chunk transaction has already rolled back.
            # Store only the exception type, avoiding sensitive details.
            try:
                with connection.transaction():
                    connection.execute(
                        """
                        UPDATE ingestion_runs
                        SET status = 'failed',
                            active = FALSE,
                            completed_at = now(),
                            error_message = %s
                        WHERE id = %s
                          AND status = 'processing'
                        """,
                        (type(error).__name__, run_id),
                    )
            except psycopg.Error:
                print(
                    "WARNING: could not record failed-run status; "
                    "the run may remain inactive and processing"
                )

            raise

        return "completed", run_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate files without connecting to PostgreSQL",
    )
    args = parser.parse_args()

    try:
        filing = load_filing()
        records = load_records(args.input, filing)
        fingerprint = content_hash(filing, records)

        print(
            f"Filing: {filing.ticker} "
            f"{filing.fiscal_year} {filing.form_type}"
        )
        print(f"Validated chunk/vector pairs: {len(records)}")
        print(f"Model: {EMBEDDING_MODEL}")
        print(f"Dimensions: {EMBEDDING_DIMENSIONS}")
        print(f"Content hash: {fingerprint}")

        if args.check_only:
            print("PASS: input validation; database not contacted")
            return 0

        status, run_id = store_records(
            filing,
            records,
            fingerprint,
        )

        print(f"Run ID: {run_id}")
        print(f"Result: {status}")

        counts = Counter(
            record.chunk.section for record in records
        )

        for section, count in counts.items():
            print(f"{section}: {count} stored chunks verified")

        if status == "skipped":
            print("PASS: identical active run reused; no inserts")
        else:
            print("PASS: chunks committed and run activated")

        return 0

    except ValidationError:
        print("FAIL: input or configuration validation failed")
        return 1

    except psycopg.Error as error:
        # Do not print connection strings or driver error details.
        print(
            f"FAIL: database operation "
            f"({type(error).__name__}, SQLSTATE={error.sqlstate})"
        )
        return 1

    except (
        ValueError,
        RuntimeError,
        OSError,
        EmbeddingError,
    ) as error:
        print(f"FAIL: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())