import json
from collections import Counter
from pathlib import Path

from app.ingestion.chunker import Chunk, build_section_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ACCESSION = "0000320193-24-000123"

# Explicit order makes the export repeatable.
SECTION_SLUGS = (
    "business",
    "risk_factors",
    "mda",
    "financial_statements",
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / f"{ACCESSION}.jsonl"
)


def load_section(slug: str) -> dict:
    """Prefer enriched sections; use parsed JSON for prose-only sections."""

    filename = f"{ACCESSION}_{slug}.json"

    enriched = PROJECT_ROOT / "data" / "enriched" / filename
    parsed = PROJECT_ROOT / "data" / "parsed" / filename

    source = enriched if enriched.exists() else parsed

    if not source.exists():
        raise FileNotFoundError(f"Missing section: {source}")

    section = json.loads(source.read_text(encoding="utf-8"))

    if section["accession_number"] != ACCESSION:
        raise ValueError(f"Unexpected filing identity in {source}")

    for block in section["blocks"]:
        if block["type"] == "table":
            if "rows" not in block or "table_number" not in block:
                raise ValueError(
                    f"Enrich tables before exporting: {source}"
                )

    return section


def main() -> None:
    chunks: list[Chunk] = []

    for slug in SECTION_SLUGS:
        section = load_section(slug)
        section_chunks = build_section_chunks(section)

        if not section_chunks:
            raise ValueError(f"No chunks produced for {slug}")

        chunks.extend(section_chunks)

        counts = Counter(
            chunk.content_type for chunk in section_chunks
        )

        print(
            f"{section['section']}: "
            f"prose={counts['prose']}, "
            f"tables={counts['table']}"
        )

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Duplicate chunk IDs in export")

    # Each line is one complete, independently readable JSON object.
    payload = "".join(
        chunk.model_dump_json() + "\n"
        for chunk in chunks
    )

    # Verify serialization before writing.
    serialized_chunks = [
        Chunk.model_validate_json(line)
        for line in payload.splitlines()
    ]

    if serialized_chunks != chunks:
        raise ValueError("Serialization changed the chunks")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Replace the previous export rather than appending duplicates.
    OUTPUT_PATH.write_text(payload, encoding="utf-8")

    # Read the actual saved file back and validate every record.
    restored = []

    with OUTPUT_PATH.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(
                    f"Unexpected blank line at {line_number}"
                )

            restored.append(Chunk.model_validate_json(line))

    if restored != chunks:
        raise ValueError("Saved chunks differ from generated chunks")

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Verified records: {len(restored)}")
    print(f"Unique chunk IDs: {len(set(chunk_ids))}")
    print("PASS: JSONL read-back matches all generated chunks")


if __name__ == "__main__":
    main()