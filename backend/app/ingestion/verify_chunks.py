import json
from collections import Counter
from pathlib import Path

from app.ingestion.chunker import (
    MAX_PROSE_WORDS,
    MAX_TABLE_WORDS,
    Chunk,
    build_section_chunks,
    generate_chunk_id,
)
from app.ingestion.tables import enrich_table, render_table_text


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ACCESSION = "0000320193-24-000123"

SECTION_SLUGS = (
    "business",
    "risk_factors",
    "mda",
    "financial_statements",
)


def require(condition: bool, message: str) -> None:
    """Keep verification active even when Python runs with -O."""
    if not condition:
        raise AssertionError(message)


def verify_merged_cells() -> None:
    """Rows connected by rowspan must remain in one piece."""

    html = """
    <table>
        <tr>
            <th>Metric</th><th>2024</th><th>2023</th>
        </tr>
        <tr>
            <td rowspan="2">Revenue</td><td>100</td><td>90</td>
        </tr>
        <tr>
            <td>110</td><td>95</td>
        </tr>
        <tr>
            <td>Profit</td><td>40</td><td>35</td>
        </tr>
    </table>
    """

    section = {
        "accession_number": ACCESSION,
        "company": "Synthetic Example",
        "ticker": "TEST",
        "fiscal_year": 2024,
        "section": "Synthetic statement",
        "citation_url": "https://example.com/filing",
        "blocks": [
            {
                "type": "table",
                "content": html,
                "table_number": 1,
                **enrich_table(html, ["Amounts in millions"]),
            },
        ],
    }

    # Force every indivisible row group above the budget.
    chunks = build_section_chunks(section, max_table_words=1)

    require(
        [
            (chunk.source_row_start, chunk.source_row_end)
            for chunk in chunks
        ] == [(1, 2), (3, 3)],
        "Rows connected by rowspan were split",
    )

    require(
        all(chunk.oversized for chunk in chunks),
        "Oversized row groups were not flagged",
    )

    require(
        "Row 2:" in chunks[0].text
        and "Row 3:" in chunks[0].text,
        "Merged row group lost content",
    )

    for chunk in chunks:
        require(
            "Amounts in millions" in chunk.text
            and "Metric" in chunk.text,
            "Repeated table context is missing",
        )

    print("PASS: merged rows, repeated context, oversized flags")


def verify_section(section: dict) -> list[Chunk]:
    chunks = build_section_chunks(section)
    blocks = section["blocks"]

    require(bool(chunks), "Section produced no chunks")

    require(
        chunks == build_section_chunks(section),
        "Repeated runs produced different chunks",
    )

    require(
        [chunk.position for chunk in chunks] == list(range(len(chunks))),
        "Chunk positions are not sequential",
    )

    require(
        len({chunk.chunk_id for chunk in chunks}) == len(chunks),
        "Duplicate chunk IDs within the section",
    )

    starts = [chunk.source_block_start for chunk in chunks]
    require(starts == sorted(starts), "Chunks are out of source order")

    for chunk in chunks:
        require(
            0 <= chunk.source_block_start
            <= chunk.source_block_end < len(blocks),
            "Invalid source block range",
        )

        for field in (
            "accession_number",
            "company",
            "ticker",
            "fiscal_year",
            "section",
            "citation_url",
        ):
            expected = section[field]
            if isinstance(expected, str):
                expected = expected.strip()

            require(
                getattr(chunk, field) == expected,
                f"Incorrect metadata: {field}",
            )

        require(
            chunk.chunk_id == generate_chunk_id(
                accession_number=chunk.accession_number,
                section=chunk.section,
                position=chunk.position,
                content_type=chunk.content_type,
                text=chunk.text,
                chunking_version=chunk.chunking_version,
            ),
            "Stored ID does not match the chunk",
        )

        require(
            Chunk.model_validate_json(chunk.model_dump_json()) == chunk,
            "JSON round trip changed a chunk",
        )

        if chunk.content_type == "prose":
            require(
                len(chunk.text.split()) <= MAX_PROSE_WORDS,
                "Prose exceeds its word budget",
            )

            require(
                all(
                    blocks[index]["type"] == "prose"
                    for index in range(
                        chunk.source_block_start,
                        chunk.source_block_end + 1,
                    )
                ),
                "Prose crossed a table boundary",
            )

        else:
            block_index = chunk.source_block_start

            require(
                chunk.source_block_end == block_index
                and blocks[block_index]["type"] == "table",
                "Table chunk has an invalid source block",
            )

            require(
                chunk.table_number == blocks[block_index]["table_number"],
                "Table identity changed",
            )

            require(
                chunk.oversized
                == (len(chunk.text.split()) > MAX_TABLE_WORDS),
                "Incorrect oversized flag",
            )

    # Check every original rendered table row survives chunking.
    for block_index, block in enumerate(blocks):
        if block["type"] != "table":
            continue

        pieces = [
            chunk
            for chunk in chunks
            if chunk.content_type == "table"
            and chunk.source_block_start == block_index
        ]

        require(bool(pieces), f"Table at block {block_index} was lost")

        actual_lines = Counter(
            line
            for piece in pieces
            for line in piece.text.splitlines()
        )

        # Before this point, rows may be repeated header context.
        body_start = min(piece.source_row_start for piece in pieces)

        for line in render_table_text(block).splitlines():
            if not line.startswith("Row "):
                require(
                    all(line in piece.text.splitlines() for piece in pieces),
                    "Table context was not repeated in every piece",
                )
                continue

            label, _ = line.split(": ", 1)
            row_index = int(label.removeprefix("Row ")) - 1

            require(actual_lines[line] >= 1, f"Lost table row: {label}")

            if row_index >= body_start:
                require(
                    actual_lines[line] == 1,
                    f"Duplicated table body row: {label}",
                )

    return chunks


def main() -> None:
    verify_merged_cells()

    all_ids = set()
    total_chunks = 0
    total_oversized = 0

    for slug in SECTION_SLUGS:
        filename = f"{ACCESSION}_{slug}.json"

        # Prefer enriched sections; prose-only sections can use parsed JSON.
        enriched = PROJECT_ROOT / "data" / "enriched" / filename
        parsed = PROJECT_ROOT / "data" / "parsed" / filename
        source = enriched if enriched.exists() else parsed

        if not source.exists():
            raise FileNotFoundError(f"Missing saved section: {source}")

        section = json.loads(source.read_text(encoding="utf-8"))

        # Fail clearly if a raw table was supplied.
        for block in section["blocks"]:
            if block["type"] == "table":
                require(
                    "rows" in block and "table_number" in block,
                    f"{source.name}: enrich tables before verification",
                )

        chunks = verify_section(section)
        ids = {chunk.chunk_id for chunk in chunks}

        require(
            not (all_ids & ids),
            "Duplicate chunk IDs across sections",
        )

        all_ids.update(ids)
        total_chunks += len(chunks)

        counts = Counter(chunk.content_type for chunk in chunks)
        oversized = sum(chunk.oversized for chunk in chunks)
        total_oversized += oversized

        print(
            f"PASS: {section['section']} | "
            f"prose={counts['prose']} | tables={counts['table']} | "
            f"oversized={oversized}"
        )

    print(
        f"\nVerified {len(SECTION_SLUGS)} sections, "
        f"{total_chunks} chunks, and unique repeatable IDs."
    )
    print(f"Oversized table pieces requiring later review: {total_oversized}")


if __name__ == "__main__":
    main()