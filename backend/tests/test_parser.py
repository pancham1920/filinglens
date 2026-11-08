import json
import tempfile
import unittest
from pathlib import Path

from bs4 import BeautifulSoup

from app.ingestion.parser import (
    extract_content_blocks,
    extract_section_elements,
    find_section_anchor,
    normalize_text,
    save_and_verify_section,
    select_section_boundaries,
)


class ParserTests(unittest.TestCase):
    def test_normalizes_nonbreaking_spaces(self):
        text = "  Item 1A.\u00a0\u00a0 Risk\n Factors  "

        self.assertEqual(
            normalize_text(text),
            "Item 1A. Risk Factors",
        )

    def test_selects_body_headings_and_excludes_end_section(self):
        soup = BeautifulSoup(
            """
            <body>
                <div><a href="#business">Item 1. Business</a></div>
                <div><a href="#risks">Item 1A. Risk Factors</a></div>

                <div id="business"></div>
                <div>Item 1. Business</div>
                <p>Business content.</p>

                <div id="risks"></div>
                <div>Item 1A. Risk Factors</div>
                <p>Risk content.</p>
            </body>
            """,
            "html.parser",
        )

        start, end = select_section_boundaries(
            soup,
            "Item 1. Business",
            "Item 1A. Risk Factors",
        )
        elements = extract_section_elements(start, end)

        self.assertEqual(find_section_anchor(start), "business")
        self.assertEqual(
            extract_content_blocks(elements),
            [{"type": "prose", "content": "Business content."}],
        )

    def test_rejects_missing_duplicate_and_reversed_headings(self):
        examples = {
            "missing": """
                <div>Item 1. Business</div>
            """,
            "duplicate": """
                <div>Item 1. Business</div>
                <div>Item 1. Business</div>
                <div>Item 1A. Risk Factors</div>
            """,
            "reversed": """
                <div>Item 1A. Risk Factors</div>
                <div>Item 1. Business</div>
            """,
        }

        for name, html in examples.items():
            with self.subTest(example=name):
                soup = BeautifulSoup(html, "html.parser")

                with self.assertRaises(ValueError):
                    select_section_boundaries(
                        soup,
                        "Item 1. Business",
                        "Item 1A. Risk Factors",
                    )

    def test_rejects_boundaries_with_different_parents(self):
        soup = BeautifulSoup(
            """
            <section><h2>Item 1. Business</h2></section>
            <section><h2>Item 1A. Risk Factors</h2></section>
            """,
            "html.parser",
        )

        start, end = select_section_boundaries(
            soup,
            "Item 1. Business",
            "Item 1A. Risk Factors",
        )

        with self.assertRaisesRegex(ValueError, "same parent"):
            extract_section_elements(start, end)

    def test_anchor_search_does_not_cross_visible_content(self):
        soup = BeautifulSoup(
            """
            <body>
                <div id="previous-section"></div>
                <p>Earlier section content.</p>
                <h2>Item 1A. Risk Factors</h2>
            </body>
            """,
            "html.parser",
        )

        self.assertIsNone(find_section_anchor(soup.h2))

    def test_removes_empty_elements_separators_and_footer(self):
        soup = BeautifulSoup(
            """
            <body>
                <div></div>
                <hr>
                <div>Apple Inc. | 2024 Form 10-K | 5</div>
                <p>Keep this paragraph.</p>
            </body>
            """,
            "html.parser",
        )

        blocks = extract_content_blocks(
            soup.body.find_all(recursive=False)
        )

        self.assertEqual(
            blocks,
            [{"type": "prose", "content": "Keep this paragraph."}],
        )

    def test_preserves_table_and_surrounding_prose_in_order(self):
        soup = BeautifulSoup(
            """
            <body>
                <ix:continuation>
                    <p>Amounts in millions.</p>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th colspan="2">2024</th>
                        </tr>
                        <tr>
                            <td>Revenue</td>
                            <td>$</td>
                            <td>
                                <ix:nonfraction scale="6">
                                    391,035
                                </ix:nonfraction>
                            </td>
                        </tr>
                    </table>
                    <p>Following explanation.</p>
                </ix:continuation>
            </body>
            """,
            "html.parser",
        )

        blocks = extract_content_blocks(
            soup.body.find_all(recursive=False)
        )

        self.assertEqual(
            [block["type"] for block in blocks],
            ["prose", "table", "prose"],
        )
        self.assertEqual(
            blocks[0]["content"],
            "Amounts in millions.",
        )
        self.assertEqual(
            blocks[2]["content"],
            "Following explanation.",
        )

        table = BeautifulSoup(blocks[1]["content"], "html.parser")

        self.assertEqual(len(table.find_all("tr")), 2)
        self.assertEqual(table.find("th", colspan=True)["colspan"], "2")
        self.assertEqual(table.find("ix:nonfraction")["scale"], "6")
        self.assertEqual(
            table.find("ix:nonfraction").get_text(strip=True),
            "391,035",
        )

    def test_saves_mixed_blocks_without_changing_content(self):
        section = {
            "section": "Financial Statements",
            "company": "Apple Inc.",
            "blocks": [
                {"type": "prose", "content": "Company’s revenue."},
                {
                    "type": "table",
                    "content": "<table><tr><td>391,035</td></tr></table>",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "parsed" / "section.json"

            save_and_verify_section(section, output_path)

            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved, section)

    def test_rejects_invalid_blocks_before_saving(self):
        invalid_blocks = [
            [],
            ["plain string instead of a block"],
            [{"type": "unknown", "content": "Text"}],
            [{"type": "prose", "content": "   "}],
            [{"type": "table", "content": "<p>No table here</p>"}],
        ]

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "invalid.json"

            for blocks in invalid_blocks:
                with self.subTest(blocks=blocks):
                    with self.assertRaises(ValueError):
                        save_and_verify_section(
                            {"blocks": blocks},
                            output_path,
                        )

                    self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()