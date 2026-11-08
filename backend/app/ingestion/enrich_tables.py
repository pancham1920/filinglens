import json
from collections import Counter
from pathlib import Path

from app.ingestion.tables import enrich_table, render_table_text


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARSED_DIR = PROJECT_ROOT / "data" / "parsed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "enriched"

ACCESSION = "0000320193-24-000123"
SECTION_SLUGS = ("mda", "financial_statements")


def enrich_section(section: dict) -> int:
    """Enrich each table while preserving section metadata and block order."""

    preceding_prose = []
    table_count = 0

    for block in section["blocks"]:
        if block["type"] == "prose":
            # Keep the three nearest prose blocks as possible table context.
            preceding_prose.append(block["content"])
            preceding_prose = preceding_prose[-3:]
            continue

        if block["type"] != "table":
            raise ValueError(
                f"Unsupported block type: {block['type']}"
            )

        table_count += 1

        # Derive metadata and positioned rows for THIS table.
        # Its original HTML remains in block["content"].
        details = enrich_table(
            block["content"],
            preceding_prose,
        )
        block.update(details)

        block["table_number"] = table_count

        # These citations currently point to the containing section.
        block["source_url"] = section["source_url"]
        block["source_anchor"] = section.get("source_anchor")
        block["citation_url"] = section["citation_url"]

        # Render only AFTER enrichment so metadata and text agree.
        block["text"] = render_table_text(block)

        # Print everything from the same current block.
        print(f"\nTable {table_count}")
        print("Title candidate:", block.get("title_candidate"))
        print("Units note:", block.get("units_note"))
        print(
            "Period headers:",
            [
                header["text"]
                for header in block["period_headers"]
            ],
        )
        print("Readable table preview:")
        print("\n".join(block["text"].splitlines()[:8]))

        # Do not automatically carry context across another table.
        # Continuation tables need explicit handling later.
        preceding_prose = []

    return table_count


def save_and_verify(section: dict, destination: Path) -> None:
    """Save enriched JSON and verify it matches the in-memory result."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(
        json.dumps(
            section,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    saved_section = json.loads(
        destination.read_text(encoding="utf-8")
    )

    if saved_section != section:
        raise ValueError(
            f"Saved JSON differs from enriched data: {destination}"
        )

    # This verifies serialization, not financial correctness.
    print("\nSaved and verified:", destination)


def main() -> None:
    total_tables = 0

    for section_slug in SECTION_SLUGS:
        filename = f"{ACCESSION}_{section_slug}.json"
        source_path = PARSED_DIR / filename
        destination = OUTPUT_DIR / filename

        # Load a fresh section so tables cannot inherit stale metadata
        # from a previous enrichment run.
        section = json.loads(
            source_path.read_text(encoding="utf-8")
        )

        print(f"\nSECTION: {section['section']}")
        print(
            "Block types:",
            dict(
                Counter(
                    block["type"]
                    for block in section["blocks"]
                )
            ),
        )

        table_count = enrich_section(section)
        save_and_verify(section, destination)

        total_tables += table_count

    print(
        f"\nFinished: {len(SECTION_SLUGS)} sections saved; "
        f"{total_tables} tables processed."
    )


if __name__ == "__main__":
    main()