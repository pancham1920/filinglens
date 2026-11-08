"""Local embedding behavior without model downloads or API calls."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.ingestion import embeddings as e
from app.ingestion.chunker import Chunk


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        tokenizer = SimpleNamespace(
            encode=lambda text, **kwargs: [0] * (len(text.split()) + 2)
        )
        self.model = SimpleNamespace(
            tokenizer=tokenizer,
            max_seq_length=512,
            encode=Mock(return_value=SimpleNamespace(
                tolist=lambda: [1.0] + [0.0] * 383
            )),
        )
        patcher = patch.object(e, "get_model", return_value=self.model)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_local_passage_returns_vector_and_requests_normalization(self):
        result = e.embed_text("A filing passage", offline=True)
        self.assertEqual(len(result), 384)
        self.model.encode.assert_called_once_with(
            "A filing passage", normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        )

    def test_query_instruction_only_applies_to_queries(self):
        e.embed_text("revenue", is_query=True)
        self.assertEqual(
            self.model.encode.call_args.args[0],
            e.QUERY_INSTRUCTION + "revenue",
        )

    def test_empty_inputs_fail_before_model_loading(self):
        for value in ("", "  ", None, 123):
            with self.subTest(value=value), self.assertRaises(ValueError):
                e.embed_text(value)
        e.get_model.assert_not_called()

    def test_exact_token_limit_is_accepted(self):
        e.embed_text(" ".join(["word"] * 510))
        self.model.encode.assert_called_once()

    def test_long_input_is_rejected_without_encoding(self):
        with self.assertRaisesRegex(e.EmbeddingError, "no text was truncated"):
            e.embed_text(" ".join(["word"] * 511))
        self.model.encode.assert_not_called()

    def test_model_limit_is_respected(self):
        self.model.max_seq_length = 5
        with self.assertRaises(e.EmbeddingError):
            e.embed_text("one two three four")
        self.model.encode.assert_not_called()

    def test_query_instruction_counts_toward_token_limit(self):
        with self.assertRaises(e.EmbeddingError):
            e.embed_text(" ".join(["word"] * 510), is_query=True)
        self.model.encode.assert_not_called()

    def test_invalid_vectors_are_rejected(self):
        vectors = [[], [0.0] * 384, [1.0] * 383,
                   [float("nan")] + [1.0] * 383,
                   [float("inf")] + [1.0] * 383,
                   [True] + [1.0] * 383]
        for vector in vectors:
            with self.subTest(vector=vector[:2]):
                self.model.encode.return_value = SimpleNamespace(
                    tolist=lambda: vector
                )
                with self.assertRaises(e.EmbeddingError):
                    e.embed_text("test")

    def test_tokenizer_truncation_is_disabled(self):
        self.model.tokenizer.encode = Mock(return_value=[1, 2, 3])
        self.assertEqual(e.token_count("test"), 3)
        self.model.tokenizer.encode.assert_called_once_with(
            "test", add_special_tokens=True, truncation=False,
        )

    def test_audit_reports_long_chunks_without_changing_export(self):
        chunk = Chunk(
            chunk_id="test-id", accession_number="0000320193-24-000123",
            company="Apple Inc.", ticker="AAPL", fiscal_year=2024,
            section="Business", content_type="prose", position=0,
            text=" ".join(["word"] * 511),
            citation_url="https://example.com/filing#business",
            source_block_start=0, source_block_end=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.jsonl"
            original = chunk.model_dump_json() + "\n"
            path.write_text(original, encoding="utf-8")
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(e.audit_chunks(path), 1)
            self.assertIn("Chunks requiring splitting: 1", output.getvalue())
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.model.encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
