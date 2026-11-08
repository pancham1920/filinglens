import json
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag


PROJECT_ROOT = Path(__file__).resolve().parents[3]

HTML_PATH = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "0000320193-24-000123.html"
)

SECTION_CONFIGS = [
    {
        "name": "Business",
        "slug": "business",
        "start": "Item 1. Business",
        "end": "Item 1A. Risk Factors",
    },
    {
        "name": "Risk Factors",
        "slug": "risk_factors",
        "start": "Item 1A. Risk Factors",
        "end": "Item 1B. Unresolved Staff Comments",
    },
    {
        "name": "MD&A",
        "slug": "mda",
        "start": (
            "Item 7. Management’s Discussion and Analysis of "
            "Financial Condition and Results of Operations"
        ),
        "end": (
            "Item 7A. Quantitative and Qualitative "
            "Disclosures About Market Risk"
        ),
    },
    {
        "name": "Financial Statements",
        "slug": "financial_statements",
        "start": (
            "Item 8. Financial Statements and Supplementary Data"
        ),
        "end": (
            "Item 9. Changes in and Disagreements with Accountants "
            "on Accounting and Financial Disclosure"
        ),
    },
]


def normalize_text(text: str) -> str:
    """Collapse whitespace, including nonbreaking spaces."""
    return " ".join(text.split())


def select_section_boundaries(
    soup: BeautifulSoup,
    start_heading: str,
    end_heading: str,
) -> tuple[Tag, Tag]:
    """Find unique, unlinked start and end headings."""
    start_text = normalize_text(start_heading).casefold()
    end_text = normalize_text(end_heading).casefold()

    if start_text == end_text:
        raise ValueError("Start and end headings must differ")

    matches: dict[str, list[Tag]] = {
        start_text: [],
        end_text: [],
    }

    for element in soup.find_all(
        ["div", "p", "h1", "h2", "h3", "h4", "h5", "h6"]
    ):
        text = normalize_text(
            element.get_text(" ", strip=True)
        ).casefold()

        if text not in matches:
            continue

        # Exclude linked navigation entries.
        if (
            element.find("a", href=True) is not None
            or element.find_parent("a", href=True) is not None
        ):
            continue

        matches[text].append(element)

    starts = matches[start_text]
    ends = matches[end_text]

    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(
            f"Expected unique boundaries for {start_heading!r}; "
            f"found {len(starts)} starts and {len(ends)} ends"
        )

    start = starts[0]
    end = ends[0]

    # Compare element identity to check document order.
    if not any(node is end for node in start.next_elements):
        raise ValueError("End heading must appear after start heading")

    return start, end


def extract_section_elements(start: Tag, end: Tag) -> list[Tag]:
    """Collect sibling elements between the two headings."""
    if start.parent is not end.parent:
        raise ValueError("Section headings must share the same parent")

    elements = []

    for node in start.next_siblings:
        # Stop before the next section's heading.
        if node is end:
            return elements

        if not isinstance(node, Tag):
            continue

        if node.name in {"script", "style", "noscript"}:
            continue

        elements.append(node)

    raise ValueError("End heading was not reached")


def find_section_anchor(start: Tag) -> str | None:
    """Find an ID on the heading or an empty preceding element."""
    if start.get("id"):
        return str(start["id"])

    for node in start.previous_siblings:
        if not isinstance(node, Tag):
            continue

        text = normalize_text(node.get_text(" ", strip=True))

        # Avoid crossing visible content into an earlier section.
        if text:
            break

        if node.get("id"):
            return str(node["id"])

    return None


def find_section_tables(elements: list[Tag]) -> list[Tag]:
    """Find table elements, including tables inside wrappers."""
    tables = []

    for element in elements:
        if element.name == "table":
            tables.append(element)

        tables.extend(element.find_all("table"))

    return tables


def extract_prose_blocks(elements: list[Tag]) -> list[str]:
    """Clean table-free elements into nonempty text blocks."""
    blocks = []

    for element in elements:
        if element.name in {"hr", "script", "style", "noscript"}:
            continue

        if element.name == "table" or element.find("table") is not None:
            raise ValueError("Use mixed-block extraction for tables")

        text = normalize_text(element.get_text(" ", strip=True))

        if not text:
            continue

        # Footer rule for the Apple FY2024 filing currently in scope.
        parts = [part.strip() for part in text.split("|")]
        if (
            len(parts) == 3
            and parts[0] == "Apple Inc."
            and parts[1] == "2024 Form 10-K"
            and parts[2].isdigit()
        ):
            continue

        blocks.append(text)

    return blocks


def extract_content_blocks(
    elements: list[Tag],
) -> list[dict[str, str]]:
    """Preserve prose and whole tables in document order."""
    blocks: list[dict[str, str]] = []

    def visit(node: Tag | NavigableString) -> None:
        if isinstance(node, Tag):
            if node.name in {"hr", "script", "style", "noscript"}:
                return

            # Capture the table once, including its nested markup.
            # Returning prevents duplicate extraction of its contents.
            if node.name == "table":
                blocks.append({
                    "type": "table",
                    "content": str(node),
                })
                return

            # Preserve prose surrounding tables inside a wrapper.
            if node.find("table") is not None:
                for child in node.children:
                    if isinstance(child, (Tag, NavigableString)):
                        visit(child)
                return

            # Table-free elements can use the existing prose cleaner.
            for text in extract_prose_blocks([node]):
                blocks.append({
                    "type": "prose",
                    "content": text,
                })

        elif type(node) is NavigableString:
            # Exact type excludes comments and other special strings.
            text = normalize_text(str(node))

            if text:
                blocks.append({
                    "type": "prose",
                    "content": text,
                })

    for element in elements:
        visit(element)

    return blocks


def save_and_verify_section(
    section: dict,
    output_path: Path,
) -> None:
    """Validate blocks, save JSON, and verify the read-back result."""
    blocks = section["blocks"]

    if not isinstance(blocks, list) or not blocks:
        raise ValueError("Section must contain a nonempty block list")

    for block in blocks:
        if not isinstance(block, dict):
            raise ValueError("Each block must be a dictionary")

        if block.get("type") not in {"prose", "table"}:
            raise ValueError("Unsupported block type")

        content = block.get("content")

        if not isinstance(content, str) or not content.strip():
            raise ValueError("Block content must be a nonempty string")

        if block["type"] == "table":
            parsed = BeautifulSoup(content, "html.parser")

            if parsed.find("table") is None:
                raise ValueError("Table block contains no table")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(section, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    saved_section = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    if saved_section != section:
        raise ValueError("Saved data differs from memory")


def main() -> None:
    # Load the cached filing once.
    html = HTML_PATH.read_bytes()
    soup = BeautifulSoup(html, "html.parser")

    # Load metadata once and match it to the cached filename.
    manifest_path = PROJECT_ROOT / "data" / "manifest.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    matching_filings = [
        entry
        for entry in manifest["filings"]
        if entry["accession_number"] == HTML_PATH.stem
    ]

    if len(matching_filings) != 1:
        raise ValueError(
            f"Expected one manifest entry for {HTML_PATH.stem}; "
            f"found {len(matching_filings)}"
        )

    filing = matching_filings[0]
    output_dir = PROJECT_ROOT / "data" / "parsed"
    saved_count = 0

    # Apply the same extraction and saving workflow to every section.
    for config in SECTION_CONFIGS:
        start, end = select_section_boundaries(
            soup,
            start_heading=config["start"],
            end_heading=config["end"],
        )

        elements = extract_section_elements(start, end)
        anchor = find_section_anchor(start)
        tables = find_section_tables(elements)
        blocks = extract_content_blocks(elements)

        citation_url = (
            f"{filing['source_url']}#{anchor}"
            if anchor
            else filing["source_url"]
        )

        section = {
            "accession_number": filing["accession_number"],
            "company": filing["company"],
            "ticker": filing["ticker"],
            "fiscal_year": filing["fiscal_year"],
            "section": config["name"],
            "source_url": filing["source_url"],
            "source_anchor": anchor,
            "citation_url": citation_url,
            "blocks": blocks,
        }

        output_path = (
            output_dir
            / f"{filing['accession_number']}_{config['slug']}.json"
        )

        save_and_verify_section(section, output_path)
        saved_count += 1

        block_counts = Counter(block["type"] for block in blocks)
        prose_blocks = [
            block["content"]
            for block in blocks
            if block["type"] == "prose"
        ]

        print(f"\nSection: {config['name']}")
        print("Raw elements:", len(elements))
        print("Table elements:", len(tables))
        print("Content block types:", dict(block_counts))
        print("Citation URL:", citation_url)
        print("Saved:", output_path)
        print("PASS: saved section matches the extracted data")
        print(
            "Prose characters:",
            sum(len(text) for text in prose_blocks),
        )

        if prose_blocks:
            print("First prose:", prose_blocks[0][:300])
            print("Last prose:", prose_blocks[-1][:300])

    print(f"\nSummary: {saved_count} sections saved and verified")


if __name__ == "__main__":
    main()