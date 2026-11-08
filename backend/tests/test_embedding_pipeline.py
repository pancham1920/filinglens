import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.ingestion import embeddings as e
from app.ingestion.chunker import Chunk
from app.ingestion.embed_chunks import embed_chunks
from app.ingestion.embedding_cache import (
    load_cached_vector,
    make_cache_key,
    save_cached_vector,
)
from app.ingestion.prepare_chunks import split_table_chunk
from app.ingestion.tables import enrich_table, render_table_text


VECTOR = [1.0] + [0.0] * 383


def make_chunk(text="test passage", **updates):
    values = {
        "chunk_id": "chunk-a",
        "accession_number": "0000320193-24-000123",
        "company": "Apple Inc.",
        "ticker": "AAPL",
        "fiscal_year": 2024,
        "section": "Business",
        "content_type": "prose",
        "position": 0,
        "text": text,
        "citation_url": "https://example.com/filing#section",
        "source_block_start": 0,
        "source_block_end": 0,
    }
    values.update(updates)
    return Chunk(**values)


class EmbeddingPipelineTests(unittest.TestCase):
    def test_cache_round_trip_and_missing_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)

            self.assertIsNone(
                load_cached_vector("missing", cache_dir=cache)
            )

            save_cached_vector(
                "passage", VECTOR, cache_dir=cache
            )

            self.assertEqual(
                load_cached_vector("passage", cache_dir=cache),
                VECTOR,
            )

    def test_cache_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            save_cached_vector("passage", VECTOR, cache_dir=cache)

            path = cache / f"{make_cache_key('passage')}.json"
            record = json.loads(path.read_text())
            record["embedding"] = [1.0]
            path.write_text(json.dumps(record))

            with self.assertRaises(e.EmbeddingError):
                load_cached_vector("passage", cache_dir=cache)

    def test_batch_preserves_vector_order(self):
        second = [0.0, 1.0] + [0.0] * 382

        model = SimpleNamespace(
            max_seq_length=512,
            tokenizer=SimpleNamespace(
                encode=lambda text, **kwargs: [1, 2, 3]
            ),
            encode=lambda texts, **kwargs: SimpleNamespace(
                tolist=lambda: [VECTOR, second]
            ),
        )

        with patch.object(e, "get_model", return_value=model):
            result = e.embed_texts(["first", "second"])

        self.assertEqual(result, [VECTOR, second])

    def test_duplicate_text_is_encoded_once_and_rerun_uses_cache(self):
        chunks = [
            make_chunk(),
            make_chunk(chunk_id="chunk-b", position=1),
        ]

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)

            with (
                patch(
                    "app.ingestion.embed_chunks.validate_inputs"
                ),
                patch(
                    "app.ingestion.embed_chunks.embed_texts",
                    return_value=[VECTOR],
                ) as generate,
                redirect_stdout(io.StringIO()),
            ):
                first = embed_chunks(chunks, cache_dir=cache)
                second = embed_chunks(chunks, cache_dir=cache)

            self.assertEqual(generate.call_count, 1)
            self.assertEqual(first, second)
            self.assertEqual(
                [record["chunk"]["chunk_id"] for record in first],
                ["chunk-a", "chunk-b"],
            )

    def test_table_split_preserves_merged_rows_and_headers(self):
        html = """
        <table>
          <tr><th>Metric</th><th>2024</th></tr>
          <tr><td rowspan="2">Revenue</td><td>100</td></tr>
          <tr><td>110</td></tr>
          <tr><td>Profit</td><td>40</td></tr>
        </table>
        """

        block = {
            "type": "table",
            "content": html,
            "table_number": 1,
            **enrich_table(html, ["Amounts in millions"]),
        }

        text = render_table_text(block)

        chunk = make_chunk(
            text,
            content_type="table",
            table_number=1,
            source_row_start=0,
            source_row_end=3,
        )

        # A synthetic budget that forces the last body row
        # into a second piece while allowing the merged group.
        count = lambda value: len(value.splitlines())
        limit = count(text) - 1

        pieces = split_table_chunk(chunk, block, count, limit)

        self.assertEqual(
            [
                (piece.source_row_start, piece.source_row_end)
                for piece in pieces
            ],
            [(1, 2), (3, 3)],
        )

        for piece in pieces:
            self.assertLessEqual(count(piece.text), limit)
            self.assertIn("Row 1:", piece.text)
            self.assertEqual(piece.citation_url, chunk.citation_url)

    def test_indivisible_table_group_is_rejected(self):
        html = """
        <table>
          <tr><th>Metric</th><th>2024</th></tr>
          <tr><td rowspan="2">Revenue</td><td>100</td></tr>
          <tr><td>110</td></tr>
        </table>
        """

        block = {
            "type": "table",
            "content": html,
            "table_number": 1,
            **enrich_table(html, ["Amounts in millions"]),
        }

        chunk = make_chunk(
            render_table_text(block),
            content_type="table",
            table_number=1,
            source_row_start=0,
            source_row_end=2,
        )

        with self.assertRaises(ValueError):
            split_table_chunk(
                chunk,
                block,
                lambda text: len(text.splitlines()),
                1,
            )


if __name__ == "__main__":
    unittest.main()