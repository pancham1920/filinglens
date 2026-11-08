import json
from pathlib import Path

from bs4 import BeautifulSoup
from app.ingestion.tables import extract_table_rows


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARSED_DIR = PROJECT_ROOT / "data" / "parsed"
ACCESSION = "0000320193-24-000123"


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    for section_slug in ["mda", "financial_statements"]:
        path = PARSED_DIR / f"{ACCESSION}_{section_slug}.json"
        section = json.loads(path.read_text(encoding="utf-8"))

        print(f"\nSECTION: {section['section']}")

        preceding_prose = []
        table_number = 0

        for block in section["blocks"]:
            if block["type"] == "prose":
                preceding_prose.append(block["content"])
                preceding_prose = preceding_prose[-3:]
                continue

            if block["type"] != "table":
                continue

            table_number += 1
            soup = BeautifulSoup(block["content"], "html.parser")
            table = soup.find("table")

            if table is None:
                raise ValueError("Table block contains no table")

            print(f"\nTABLE {table_number}")

            structured_rows = extract_table_rows(block["content"])

            print("Structured row preview:")
            nonempty_rows = [
                row
                for row in structured_rows
                if any(cell["text"] for cell in row)
            ]

            for row in nonempty_rows[:3]:
                print(row)

            print("Nearby preceding text:")

            for text in preceding_prose:
                print(" ", text[:250])

            print("First three nonempty rows:")
            shown = 0

            for row in table.find_all("tr"):
                # Exclude rows belonging to nested tables.
                if row.find_parent("table") is not table:
                    continue

                cells = [
                    normalize_text(cell.get_text(" ", strip=True))
                    for cell in row.find_all(
                        ["th", "td"], recursive=False
                    )
                ]

                if not any(cells):
                    continue

                print(" ", cells)
                shown += 1

                if shown == 3:
                    break

            # Avoid reusing context from before this table.
            preceding_prose = []


if __name__ == "__main__":
    main()