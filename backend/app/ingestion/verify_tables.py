import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "enriched"
    / "0000320193-24-000123_financial_statements.json"
)

EXPECTED = {
    "Total net sales": {
        "September 28, 2024": "391,035",
        "September 30, 2023": "383,285",
        "September 24, 2022": "394,328",
    },
    "Research and development": {
        "September 28, 2024": "31,370",
        "September 30, 2023": "29,915",
        "September 24, 2022": "26,251",
    },
}


def main() -> None:
    section = json.loads(
        SOURCE_PATH.read_text(encoding="utf-8")
    )

    # Select the actual statement rather than another table
    # that happens to contain the same metric.
    matches = [
        block
        for block in section["blocks"]
        if block["type"] == "table"
        and block.get("title_candidate")
        == "CONSOLIDATED STATEMENTS OF OPERATIONS"
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one operations statement; found {len(matches)}"
        )

    table = matches[0]

    # Verify that the table retains its reported scale.
    units_note = table.get("units_note") or ""

    if "in millions" not in units_note.casefold():
        raise ValueError("Expected an 'in millions' units note")

    checked = 0

    for metric, expected_periods in EXPECTED.items():
        # Locate the row using its metric label.
        matching_rows = [
            row
            for row in table["rows"]
            if any(
                cell["text"].casefold() == metric.casefold()
                for cell in row
            )
        ]

        if len(matching_rows) != 1:
            raise ValueError(
                f"Expected one row for {metric}; "
                f"found {len(matching_rows)}"
            )

        row = matching_rows[0]

        for period, expected_value in expected_periods.items():
            # Currency symbols may occupy separate cells.
            # Select the number cells associated with this period.
            values = [
                cell["text"]
                for cell in row
                if cell.get("period_header") == period
                and cell["text"]
                and cell["text"] != "$"
            ]

            if values != [expected_value]:
                raise ValueError(
                    f"{metric}, {period}: "
                    f"expected {[expected_value]}, got {values}"
                )

            # Confirm that the verified value also survives rendering.
            if expected_value not in table.get("text", ""):
                raise ValueError(
                    f"Value missing from readable text: {expected_value}"
                )

            checked += 1
            print(
                f"PASS: {metric} | {period} | {expected_value}"
            )

    print(f"\nVerified: {checked}/6 values and their period mappings.")
    print("Reported scale:", units_note)
    print("Citation:", table["citation_url"])


if __name__ == "__main__":
    main()