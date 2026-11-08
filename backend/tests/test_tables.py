import unittest

from app.ingestion.tables import (
    enrich_table,
    extract_table_rows,
    position_table_cells,
    render_table_text,
)


class TableTests(unittest.TestCase):
    def test_extracts_empty_cells_and_spans(self):
        """Empty cells and merged-cell widths must survive extraction."""
        html = """
        <table>
            <tr>
                <td colspan="3"></td>
                <td colspan="3">2024</td>
            </tr>
        </table>
        """

        rows = extract_table_rows(html)

        self.assertEqual(len(rows[0]), 2)
        self.assertEqual(rows[0][0]["text"], "")
        self.assertEqual(rows[0][0]["colspan"], 3)
        self.assertEqual(rows[0][1]["text"], "2024")

    def test_maps_values_to_periods_using_columns(self):
        """Different cell indexes can still align physically."""
        html = """
        <table>
            <tr>
                <td colspan="3"></td>
                <td colspan="3">2024</td>
                <td colspan="3">2023</td>
            </tr>
            <tr>
                <td colspan="3">Net income</td>
                <td>$</td>
                <td>93,736</td>
                <td></td>
                <td>$</td>
                <td>96,995</td>
                <td></td>
            </tr>
        </table>
        """

        table = enrich_table(html, [])
        cells = {
            cell["text"]: cell
            for cell in table["rows"][1]
            if cell["text"]
        }

        self.assertEqual(cells["93,736"]["column_start"], 4)
        self.assertEqual(cells["93,736"]["period_header"], "2024")

        self.assertEqual(cells["96,995"]["column_start"], 7)
        self.assertEqual(cells["96,995"]["period_header"], "2023")

        self.assertIsNone(cells["Net income"]["period_header"])

    def test_rowspan_reserves_columns_in_following_row(self):
        """A cell spanning two rows occupies its column in both."""
        html = """
        <table>
            <tr>
                <td rowspan="2">Metric</td>
                <td>2024</td>
            </tr>
            <tr>
                <td>31,370</td>
            </tr>
        </table>
        """

        rows = position_table_cells(extract_table_rows(html))

        self.assertEqual(rows[0][0]["column_start"], 0)
        self.assertEqual(rows[1][0]["column_start"], 1)

    def test_missing_context_is_not_invented(self):
        """A table without periods or units must retain that uncertainty."""
        html = """
        <table>
            <tr>
                <td>Description</td>
                <td>Supply chain risks</td>
            </tr>
        </table>
        """

        table = enrich_table(html, [])
        text = render_table_text(table)

        self.assertIsNone(table["title_candidate"])
        self.assertIsNone(table["units_note"])
        self.assertEqual(table["period_headers"], [])
        self.assertIn("Units note: not detected", text)
        self.assertIn("Period headers: not detected", text)
        self.assertIn("Supply chain risks", text)

    def test_rendering_preserves_values_and_unmapped_changes(self):
        """Rendering must retain numbers, signs, and change percentages."""
        html = """
        <table>
            <tr>
                <td>Metric</td>
                <td>2024</td>
                <td>Change</td>
                <td>2023</td>
            </tr>
            <tr>
                <td>Revenue</td>
                <td>1,234.50</td>
                <td>(5)%</td>
                <td>1,299.47</td>
            </tr>
        </table>
        """

        table = enrich_table(html, ["Amounts in millions"])
        text = render_table_text(table)

        self.assertIn("2024: 1,234.50", text)
        self.assertIn("2023: 1,299.47", text)
        self.assertIn("(5)%", text)
        self.assertIn("Amounts in millions", text)

        # The change column must not receive an invented year.
        change_cell = table["rows"][1][2]
        self.assertIsNone(change_cell["period_header"])


if __name__ == "__main__":
    unittest.main()