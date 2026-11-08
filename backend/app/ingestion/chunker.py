from typing import Literal
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.ingestion.tables import render_table_text
import json
from uuid import NAMESPACE_URL, uuid5


# Change this when chunking rules change.
# Later, we will include it when generating stable chunk IDs.
CHUNKING_VERSION = "v1"


class Chunk(BaseModel):
    """One retrievable piece of a filing, with source metadata."""

    # Reject unexpected fields so spelling mistakes fail visibly.
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    # Generated deterministically in milestone 2.4.4.
    chunk_id: str = Field(min_length=1)

    # Filing and company metadata support retrieval filters.
    accession_number: str = Field(
        pattern=r"^\d{10}-\d{2}-\d{6}$"
    )
    company: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    fiscal_year: int = Field(ge=1900, le=2100)

    section: str = Field(min_length=1)
    content_type: Literal["prose", "table"]

    # Zero-based order within this filing section.
    position: int = Field(ge=0)

    # Prose text or the readable table text created in 2.3.
    text: str = Field(min_length=1)

    # Currently points to the containing filing section.
    citation_url: str = Field(min_length=1)

    # Zero-based, inclusive indexes into the section's original blocks.
    # Overlapping chunks can reference some of the same blocks.
    source_block_start: int = Field(ge=0)
    source_block_end: int = Field(ge=0)

    # Table chunks retain their parent table's number.
    # Multiple chunks may come from one large table.
    table_number: int | None = Field(default=None, ge=1)

    # Original table row range, excluding repeated header context.
    source_row_start: int | None = Field(default=None, ge=0)
    source_row_end: int | None = Field(default=None, ge=0)

    # Preserve this flag for later embedding-limit checks.
    oversized: bool = False

    chunking_version: str = Field(
        default=CHUNKING_VERSION,
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_source_metadata(self) -> "Chunk":
        """Reject inconsistent source ranges and content types."""

        if self.source_block_end < self.source_block_start:
            raise ValueError(
                "source_block_end must be >= source_block_start"
            )

        if self.content_type == "table" and self.table_number is None:
            raise ValueError(
                "Table chunks must include table_number"
            )

        if self.content_type == "prose" and self.table_number is not None:
            raise ValueError(
                "Prose chunks must not include table_number"
            )
        
        has_row_start = self.source_row_start is not None
        has_row_end = self.source_row_end is not None

        if has_row_start != has_row_end:
            raise ValueError(
                "source_row_start and source_row_end must be supplied together"
            )

        if (
            self.source_row_start is not None
            and self.source_row_end is not None
            and self.source_row_end < self.source_row_start
        ):
            raise ValueError(
                "source_row_end must be >= source_row_start"
            )

        if self.content_type == "prose":
            if has_row_start or self.oversized:
                raise ValueError(
                    "Prose chunks must not contain table row metadata "
                    "or the oversized table flag"
                )
        return self


    
MAX_PROSE_WORDS = 350
PROSE_OVERLAP_WORDS = 50


@dataclass(frozen=True)
class ProsePiece:
    """A draft prose chunk before adding IDs and filing metadata."""

    text: str
    source_block_start: int
    source_block_end: int


def _chunk_prose_run(
    words: list[tuple[str, int]],
    max_words: int,
    overlap_words: int,
) -> list[ProsePiece]:
    """Split consecutive prose while remembering each word's source block."""

    pieces: list[ProsePiece] = []
    start = 0

    while start < len(words):
        # Take at most max_words from the current starting position.
        end = min(start + max_words, len(words))
        window = words[start:end]

        pieces.append(
            ProsePiece(
                text=" ".join(word for word, _ in window),
                source_block_start=window[0][1],
                source_block_end=window[-1][1],
            )
        )

        # Stop once the final word has been included.
        # This avoids producing an extra chunk containing only overlap.
        if end == len(words):
            break

        # Reuse the last overlap_words in the next chunk.
        start = end - overlap_words

    return pieces


def chunk_prose_blocks(
    blocks: list[dict],
    max_words: int = MAX_PROSE_WORDS,
    overlap_words: int = PROSE_OVERLAP_WORDS,
) -> list[ProsePiece]:
    """Chunk one section's prose without crossing table boundaries."""

    if max_words <= 0:
        raise ValueError("max_words must be greater than zero")

    # Overlap must be smaller than the chunk so we always move forward.
    if not 0 <= overlap_words < max_words:
        raise ValueError(
            "overlap_words must be >= 0 and smaller than max_words"
        )

    pieces: list[ProsePiece] = []

    # Each tuple stores a word and its original section block index.
    current_run: list[tuple[str, int]] = []

    for block_index, block in enumerate(blocks):
        block_type = block.get("type")

        if block_type == "prose":
            content = block.get("content")

            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Prose block {block_index} needs nonempty content"
                )

            current_run.extend(
                (word, block_index)
                for word in content.split()
            )

        elif block_type == "table":
            # Finish prose before the table.
            # Prose after the table starts a separate run.
            pieces.extend(
                _chunk_prose_run(
                    current_run,
                    max_words,
                    overlap_words,
                )
            )
            current_run = []

        else:
            raise ValueError(
                f"Unsupported block type at index {block_index}: "
                f"{block_type!r}"
            )

    # Save any prose remaining after the final block.
    pieces.extend(
        _chunk_prose_run(
            current_run,
            max_words,
            overlap_words,
        )
    )

    return pieces

MAX_TABLE_WORDS = 350


@dataclass(frozen=True)
class TablePiece:
    """A draft table chunk with its original source location."""

    text: str
    source_block_start: int
    source_block_end: int
    table_number: int

    # Zero-based, inclusive original row indexes.
    # Repeated header rows are additional context.
    source_row_start: int
    source_row_end: int

    # True when an indivisible row group exceeds the budget.
    oversized: bool


def _table_row_groups(rows: list[list[dict]]) -> list[tuple[int, int]]:
    """Group rows connected by rowspan so they stay together."""

    groups = []
    start = 0

    while start < len(rows):
        end = start
        current = start

        # The endpoint can grow when another merged cell is encountered.
        while current <= end:
            for cell in rows[current]:
                rowspan = cell["rowspan"]

                if rowspan < 1:
                    raise ValueError("Expected positive enriched rowspan")

                end = max(end, current + rowspan - 1)

            if end >= len(rows):
                raise ValueError("Rowspan extends beyond the table")

            current += 1

        groups.append((start, end))
        start = end + 1

    return groups


def _chunk_table_block(
    block: dict,
    block_index: int,
    max_words: int,
) -> list[TablePiece]:
    """Split one enriched table without splitting individual row groups."""

    rows = block.get("rows")
    table_number = block.get("table_number")

    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"Table block {block_index} needs enriched, nonempty rows"
        )

    if type(table_number) is not int or table_number < 1:
        raise ValueError(
            f"Table block {block_index} needs a positive table_number"
        )

    # Render the whole table first. Its row labels therefore retain
    # their original numbers even when we split the text afterward.
    full_text = render_table_text(block)

    context_lines = []
    row_lines = {}

    for line in full_text.splitlines():
        if line.startswith("Row "):
            label, _ = line.split(": ", 1)
            row_index = int(label.removeprefix("Row ")) - 1
            row_lines[row_index] = line
        else:
            context_lines.append(line)

    if not row_lines:
        raise ValueError(
            f"Table block {block_index} has no nonempty rows"
        )

    groups = _table_row_groups(rows)

    # Repeat everything through the detected period-header row.
    # This also retains any introductory rows before that header.
    header_end = max(
        (
            header["row_index"]
            for header in block.get("period_headers", [])
        ),
        default=-1,
    )

    # Also recognize leading rows made entirely of HTML <th> cells.
    for row_index, row in enumerate(rows):
        nonempty_cells = [cell for cell in row if cell["text"]]

        if not nonempty_cells:
            continue

        if all(cell["is_header"] for cell in nonempty_cells):
            header_end = max(header_end, row_index)
        else:
            break

    # If a header has a rowspan, repeat its complete connected group.
    for start, end in groups:
        if start <= header_end:
            header_end = max(header_end, end)

    def render_range(start: int, end: int) -> str:
        indexes = list(range(header_end + 1))
        indexes.extend(range(max(start, header_end + 1), end + 1))

        lines = context_lines + [
            row_lines[index]
            for index in indexes
            if index in row_lines
        ]

        return "\n".join(lines)

    def make_piece(start: int, end: int) -> TablePiece:
        text = render_range(start, end)

        return TablePiece(
            text=text,
            source_block_start=block_index,
            source_block_end=block_index,
            table_number=table_number,
            source_row_start=start,
            source_row_end=end,
            oversized=len(text.split()) > max_words,
        )

    # Small tables stay intact.
    if len(full_text.split()) <= max_words:
        return [make_piece(0, len(rows) - 1)]

    body_groups = [
        (start, end)
        for start, end in groups
        if start > header_end
    ]

    # A table containing only headers cannot be split further.
    if not body_groups:
        return [make_piece(0, len(rows) - 1)]

    pieces = []
    current_start, current_end = body_groups[0]

    for start, end in body_groups[1:]:
        candidate = render_range(current_start, end)

        if len(candidate.split()) <= max_words:
            current_end = end
        else:
            pieces.append(make_piece(current_start, current_end))
            current_start, current_end = start, end

    pieces.append(make_piece(current_start, current_end))

    return pieces


def chunk_table_blocks(
    blocks: list[dict],
    max_words: int = MAX_TABLE_WORDS,
) -> list[TablePiece]:
    """Chunk one section's enriched tables in their original order."""

    if type(max_words) is not int or max_words <= 0:
        raise ValueError("max_words must be a positive integer")

    pieces = []

    for block_index, block in enumerate(blocks):
        block_type = block.get("type")

        if block_type == "prose":
            continue

        if block_type != "table":
            raise ValueError(
                f"Unsupported block type at index {block_index}: "
                f"{block_type!r}"
            )

        pieces.extend(
            _chunk_table_block(block, block_index, max_words)
        )

    return pieces

def generate_chunk_id(
    *,
    accession_number: str,
    section: str,
    position: int,
    content_type: Literal["prose", "table"],
    text: str,
    chunking_version: str = CHUNKING_VERSION,
) -> str:
    """Generate a repeatable ID from a chunk's identity and content."""

    # Match the Chunk model's trimming of surrounding whitespace.
    accession_number = accession_number.strip()
    section = section.strip()
    text = text.strip()
    chunking_version = chunking_version.strip()

    if not all(
        [accession_number, section, text, chunking_version]
    ):
        raise ValueError("Chunk identity fields must not be empty")

    if type(position) is not int or position < 0:
        raise ValueError("position must be a nonnegative integer")

    if content_type not in {"prose", "table"}:
        raise ValueError("content_type must be 'prose' or 'table'")

    identity = {
        "accession_number": accession_number,
        "section": section,
        "position": position,
        "content_type": content_type,
        "text": text,
        "chunking_version": chunking_version,
    }

    # Canonical JSON makes serialization repeatable and unambiguous.
    serialized = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    # Prefix distinguishes our IDs from other applications using UUID5.
    return str(
        uuid5(
            NAMESPACE_URL,
            f"filinglens:chunk:{serialized}",
        )
    )

@dataclass(frozen=True)
class OrderedPiece:
    """A prose or table piece with its position in the section."""

    position: int
    piece: ProsePiece | TablePiece

    @property
    def content_type(self) -> Literal["prose", "table"]:
        return "table" if isinstance(self.piece, TablePiece) else "prose"

def chunk_section_blocks(
    blocks: list[dict],
    *,
    max_prose_words: int = MAX_PROSE_WORDS,
    prose_overlap_words: int = PROSE_OVERLAP_WORDS,
    max_table_words: int = MAX_TABLE_WORDS,
) -> list[OrderedPiece]:
    """Combine one section's prose and tables in source order."""

    prose_pieces = chunk_prose_blocks(
        blocks,
        max_words=max_prose_words,
        overlap_words=prose_overlap_words,
    )

    table_pieces = chunk_table_blocks(
        blocks,
        max_words=max_table_words,
    )

    combined: list[ProsePiece | TablePiece] = [
        *prose_pieces,
        *table_pieces,
    ]

    # Python's sort is stable: pieces with the same source block
    # retain the order produced by their respective chunker.
    combined.sort(key=lambda piece: piece.source_block_start)

    return [
        OrderedPiece(position=position, piece=piece)
        for position, piece in enumerate(combined)
    ]

def build_section_chunks(
    section: dict,
    *,
    max_prose_words: int = MAX_PROSE_WORDS,
    prose_overlap_words: int = PROSE_OVERLAP_WORDS,
    max_table_words: int = MAX_TABLE_WORDS,
    chunking_version: str = CHUNKING_VERSION,
) -> list[Chunk]:
    """Build complete chunks from one parsed/enriched filing section."""

    required_fields = (
        "accession_number",
        "company",
        "ticker",
        "fiscal_year",
        "section",
        "citation_url",
        "blocks",
    )

    missing = [
        field
        for field in required_fields
        if field not in section
    ]

    if missing:
        raise ValueError(
            f"Section is missing required fields: {', '.join(missing)}"
        )

    if not isinstance(section["blocks"], list):
        raise ValueError("Section blocks must be a list")

    ordered = chunk_section_blocks(
        section["blocks"],
        max_prose_words=max_prose_words,
        prose_overlap_words=prose_overlap_words,
        max_table_words=max_table_words,
    )

    chunks = []

    for item in ordered:
        piece = item.piece
        is_table = isinstance(piece, TablePiece)

        chunk_id = generate_chunk_id(
            accession_number=section["accession_number"],
            section=section["section"],
            position=item.position,
            content_type=item.content_type,
            text=piece.text,
            chunking_version=chunking_version,
        )

        chunk = Chunk(
            chunk_id=chunk_id,
            accession_number=section["accession_number"],
            company=section["company"],
            ticker=section["ticker"],
            fiscal_year=section["fiscal_year"],
            section=section["section"],
            content_type=item.content_type,
            position=item.position,
            text=piece.text,
            citation_url=section["citation_url"],
            source_block_start=piece.source_block_start,
            source_block_end=piece.source_block_end,
            table_number=piece.table_number if is_table else None,
            source_row_start=piece.source_row_start if is_table else None,
            source_row_end=piece.source_row_end if is_table else None,
            oversized=piece.oversized if is_table else False,
            chunking_version=chunking_version,
        )

        chunks.append(chunk)

    return chunks