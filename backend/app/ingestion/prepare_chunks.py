"""Prepare the Apple sample for BGE without changing the 2.4 export."""

from collections import Counter
from functools import lru_cache

from app.db.connection import PROJECT_ROOT
from app.ingestion.chunker import (
    Chunk,
    _table_row_groups,
    build_section_chunks,
    generate_chunk_id,
)
from app.ingestion.downloader import save_atomically
from app.ingestion.embeddings import (
    MAX_INPUT_TOKENS,
    get_model,
    token_count,
)
from app.ingestion.export_chunks import (
    ACCESSION,
    SECTION_SLUGS,
    load_section,
)
from app.ingestion.tables import render_table_text


PREPARED_VERSION = "v2-bge-token-aware"

INPUT_PATH = (
    PROJECT_ROOT / "data/chunks" / f"{ACCESSION}.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/prepared_chunks"
    / f"{ACCESSION}.jsonl"
)


def split_table_chunk(
    chunk: Chunk,
    block: dict,
    count,
    limit: int,
) -> list[Chunk]:
    """Split one original table chunk between complete row groups."""

    if count(chunk.text) <= limit:
        return [chunk]

    if block["table_number"] != chunk.table_number:
        raise ValueError("Table identity does not match source")

    rows = block["rows"]
    groups = _table_row_groups(rows)

    context = []
    row_lines = {}

    for line in render_table_text(block).splitlines():
        if line.startswith("Row "):
            label, _ = line.split(": ", 1)
            index = int(label.removeprefix("Row ")) - 1
            row_lines[index] = line
        else:
            context.append(line)

    header_end = max(
        (
            header["row_index"]
            for header in block.get("period_headers", [])
        ),
        default=-1,
    )

    # Include leading HTML header rows.
    for index, row in enumerate(rows):
        cells = [cell for cell in row if cell["text"]]

        if not cells:
            continue

        if all(cell["is_header"] for cell in cells):
            header_end = max(header_end, index)
        else:
            break

    # Headers connected by rowspan must stay together.
    for start, end in groups:
        if start <= header_end:
            header_end = max(header_end, end)

    if (
        chunk.source_row_start is None
        or chunk.source_row_end is None
    ):
        raise ValueError("Table chunk is missing source row ranges")

    first = max(chunk.source_row_start, header_end + 1)
    last = chunk.source_row_end

    body_groups = [
        (start, end)
        for start, end in groups
        if end >= first and start <= last
    ]

    if not body_groups:
        raise ValueError(
            f"Table {chunk.table_number}: "
            "oversized header-only content cannot be split"
        )

    if any(
        start < first or end > last
        for start, end in body_groups
    ):
        raise ValueError("Source range cuts through merged rows")

    def render(start: int, end: int) -> str:
        indexes = list(range(header_end + 1))
        indexes.extend(range(start, end + 1))

        return "\n".join(
            context
            + [
                row_lines[index]
                for index in indexes
                if index in row_lines
            ]
        )

    def make_piece(start: int, end: int) -> Chunk:
        text = render(start, end)

        if count(text) > limit:
            raise ValueError(
                f"Table {chunk.table_number}, rows "
                f"{start}–{end}: indivisible group exceeds "
                f"{limit} tokens"
            )

        return chunk.model_copy(
            update={
                "text": text,
                "source_row_start": start,
                "source_row_end": end,
                "oversized": False,
            }
        )

    # Verify that every indivisible group can fit with context.
    for start, end in body_groups:
        make_piece(start, end)

    pieces = []
    current_start, current_end = body_groups[0]

    for start, end in body_groups[1:]:
        if count(render(current_start, end)) <= limit:
            current_end = end
        else:
            pieces.append(make_piece(current_start, current_end))
            current_start, current_end = start, end

    pieces.append(make_piece(current_start, current_end))

    # Every original body row must occur exactly once.
    expected = Counter(
        row_lines[index]
        for index in range(first, last + 1)
        if index in row_lines
    )

    actual = Counter(
        line
        for piece in pieces
        for line in piece.text.splitlines()
        if line.startswith("Row ")
        and first
        <= int(line.split(": ", 1)[0].removeprefix("Row ")) - 1
        <= last
    )

    if actual != expected:
        raise ValueError("Splitting lost or duplicated table body rows")

    return pieces


def prepare_file() -> list[Chunk]:
    original = [
        Chunk.model_validate_json(line)
        for line in INPUT_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    sections = [load_section(slug) for slug in SECTION_SLUGS]

    # Ensure the saved export still matches the structured sources.
    rebuilt = [
        chunk
        for section in sections
        for chunk in build_section_chunks(section)
    ]

    if not original or original != rebuilt:
        raise ValueError(
            "Original export differs from source sections. "
            "Verify the 2.4 export before continuing."
        )

    by_name = {
        section["section"].strip(): section
        for section in sections
    }

    model = get_model(offline=True)
    limit = min(MAX_INPUT_TOKENS, model.max_seq_length)

    @lru_cache(maxsize=8192)
    def count(text: str) -> int:
        return token_count(text, offline=True)

    prepared = []
    split_parents = 0

    for chunk in original:
        if count(chunk.text) <= limit:
            prepared.append(chunk)
            continue

        # All prose in the audited Apple sample already fits.
        if chunk.content_type != "table":
            raise ValueError(
                f"Unexpected oversized prose: "
                f"{chunk.section} / {chunk.position}"
            )

        if chunk.source_block_start != chunk.source_block_end:
            raise ValueError("Expected one source block per table")

        section = by_name[chunk.section]
        block = section["blocks"][chunk.source_block_start]

        pieces = split_table_chunk(chunk, block, count, limit)
        prepared.extend(pieces)
        split_parents += 1

    # New section-local positions and IDs describe the prepared export.
    positions = Counter()
    result = []

    for chunk in prepared:
        position = positions[chunk.section]
        positions[chunk.section] += 1

        updated = chunk.model_copy(
            update={
                "position": position,
                "chunking_version": PREPARED_VERSION,
            }
        )

        updated = updated.model_copy(
            update={
                "chunk_id": generate_chunk_id(
                    accession_number=updated.accession_number,
                    section=updated.section,
                    position=updated.position,
                    content_type=updated.content_type,
                    text=updated.text,
                    chunking_version=PREPARED_VERSION,
                )
            }
        )

        result.append(
            Chunk.model_validate(updated.model_dump())
        )

    if len({chunk.chunk_id for chunk in result}) != len(result):
        raise ValueError("Duplicate prepared chunk IDs")

    for chunk in result:
        if count(chunk.text) > limit:
            raise ValueError("Prepared chunk exceeds token limit")

    payload = "".join(
        chunk.model_dump_json() + "\n"
        for chunk in result
    )

    save_atomically(OUTPUT_PATH, payload.encode("utf-8"))

    restored = [
        Chunk.model_validate_json(line)
        for line in OUTPUT_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    if restored != result:
        raise ValueError("Prepared export read-back failed")

    print(f"Original chunks: {len(original)}")
    print(f"Original chunks split: {split_parents}")
    print(f"Prepared chunks: {len(result)}")
    print(f"Maximum tokens: {max(count(c.text) for c in result)}")
    print("PASS: prepared chunks fit the model")

    return result


def main() -> None:
    prepare_file()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()