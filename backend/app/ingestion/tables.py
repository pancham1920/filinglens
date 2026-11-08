from bs4 import BeautifulSoup

from app.ingestion.parser import normalize_text
import re
from copy import deepcopy


def extract_table_rows(table_html: str) -> list[list[dict]]:
    """Extract cells without discarding empty cells or merged-cell spans."""
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table")

    if table is None:
        raise ValueError("HTML contains no table")

    rows = []

    for row in table.find_all("tr"):
        # Exclude rows belonging to a nested table.
        if row.find_parent("table") is not table:
            continue

        cells = []

        for cell in row.find_all(["th", "td"], recursive=False):
            # A merged cell may occupy several columns or rows.
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))

            # HTML permits rowspan=0: span to the end of the row group.
            if colspan < 1 or rowspan < 0:
                raise ValueError("Invalid table cell span")

            cells.append({
                "text": normalize_text(
                    cell.get_text(" ", strip=True)
                ),
                "is_header": cell.name == "th",
                "colspan": colspan,
                "rowspan": rowspan,
            })

        # Retain empty cells and rows to preserve table structure.
        rows.append(cells)

    return rows

def position_table_cells(rows: list[list[dict]]) -> list[list[dict]]:
    """Calculate zero-based column positions, accounting for spans."""
    positioned_rows = deepcopy(rows)

    # Coordinates occupied by cells extending across columns or rows.
    occupied = set()

    for row_index, row in enumerate(positioned_rows):
        column = 0

        for cell in row:
            # Skip positions occupied by a cell from an earlier row.
            while (row_index, column) in occupied:
                column += 1

            colspan = cell["colspan"]
            rowspan = cell["rowspan"]

            # rowspan=0 requires row-group information that our current
            # row extractor does not retain. Fail rather than misalign.
            if rowspan == 0:
                raise ValueError(
                    "rowspan=0 requires row-group support"
                )

            column_end = column + colspan - 1
            row_end = row_index + rowspan - 1

            if row_end >= len(positioned_rows):
                raise ValueError("Cell rowspan extends beyond the table")

            coordinates = {
                (r, c)
                for r in range(row_index, row_end + 1)
                for c in range(column, column_end + 1)
            }

            if coordinates & occupied:
                raise ValueError("Overlapping merged table cells")

            occupied.update(coordinates)

            cell["row_index"] = row_index
            cell["column_start"] = column
            cell["column_end"] = column_end

            column = column_end + 1

    return positioned_rows


def find_period_headers(rows: list[list[dict]]) -> list[dict]:
    """Find the first row containing standalone year/date headers."""
    period_pattern = re.compile(
        r"(?:(?:January|February|March|April|May|June|July|"
        r"August|September|October|November|December)"
        r"\s+\d{1,2},?\s+)?"
        r"(?P<year>(?:19|20)\d{2})",
        re.IGNORECASE,
    )

    for row in rows:
        headers = []

        for cell in row:
            match = period_pattern.fullmatch(cell["text"])

            if match is None:
                continue

            headers.append({
                "text": cell["text"],
                "year": int(match.group("year")),
                "row_index": cell["row_index"],
                "column_start": cell["column_start"],
                "column_end": cell["column_end"],
            })

        if headers:
            return headers

    return []


def associate_period_headers(
    rows: list[list[dict]],
    headers: list[dict],
) -> None:
    """Attach a header only when a cell fits fully beneath one header."""
    for row in rows:
        for cell in row:
            cell["period_header"] = None

            if not cell["text"]:
                continue

            matches = [
                header
                for header in headers
                if cell["row_index"] > header["row_index"]
                and cell["column_start"] >= header["column_start"]
                and cell["column_end"] <= header["column_end"]
            ]

            if len(matches) == 1:
                cell["period_header"] = matches[0]["text"]


def extract_table_context(preceding_prose: list[str]) -> dict:
    """Keep source context and identify conservative context candidates."""
    units_index = None

    for index, text in enumerate(preceding_prose):
        # These patterns cover the examples we inspected.
        if re.search(
            r"\bin (?:millions|thousands)\b",
            text,
            re.IGNORECASE,
        ):
            units_index = index

    units_note = (
        preceding_prose[units_index]
        if units_index is not None
        else None
    )

    title_candidate = None

    # In the inspected examples, the title immediately precedes
    # the explanatory units paragraph.
    if units_index is not None and units_index > 0:
        possible_title = preceding_prose[units_index - 1]

        if len(possible_title) <= 200:
            title_candidate = possible_title

    return {
        "preceding_prose": list(preceding_prose),
        "title_candidate": title_candidate,
        "units_note": units_note,
    }


def enrich_table(
    table_html: str,
    preceding_prose: list[str],
) -> dict:
    """Combine cell positions, period headers, and surrounding context."""
    rows = position_table_cells(extract_table_rows(table_html))
    headers = find_period_headers(rows)

    associate_period_headers(rows, headers)
    context = extract_table_context(preceding_prose)

    return {
        **context,
        "period_headers": headers,
        "period_header_status": (
            "detected_candidate" if headers else "missing"
        ),
        "rows": rows,
    }

def render_table_text(table: dict) -> str:
    """Create readable text without changing the reported values."""

    lines = []

    # Keep the title and complete units note beside the table.
    if table.get("title_candidate"):
        lines.append(f"Title: {table['title_candidate']}")

    if table.get("units_note"):
        lines.append(f"Units note: {table['units_note']}")
    else:
        lines.append("Units note: not detected")

    if not table.get("period_headers"):
        lines.append("Period headers: not detected")

    for row_number, row in enumerate(table["rows"], start=1):
        groups = []

        for cell in row:
            text = cell["text"]

            # Empty cells already contributed to column positioning.
            # They do not need to appear in the readable text.
            if not text:
                continue

            period = cell.get("period_header")

            # Join adjacent nonempty cells belonging to the same period.
            # For example, "$" and "93,736" become "$ 93,736".
            if (
                groups
                and period is not None
                and groups[-1]["period"] == period
            ):
                groups[-1]["texts"].append(text)
                groups[-1]["column_end"] = cell["column_end"]
            else:
                groups.append(
                    {
                        "period": period,
                        "column_start": cell["column_start"],
                        "column_end": cell["column_end"],
                        "texts": [text],
                    }
                )

        # Skip rows containing only empty cells.
        if not groups:
            continue

        parts = []

        for group in groups:
            text = " ".join(group["texts"])

            if group["period"] is not None:
                label = group["period"]
            else:
                # Preserve unmatched cells, including row labels and
                # change percentages, without inventing a period.
                label = (
                    f"columns {group['column_start']}"
                    f"–{group['column_end']}"
                )

            parts.append(f"{label}: {text}")

        lines.append(f"Row {row_number}: " + " | ".join(parts))

    return "\n".join(lines)