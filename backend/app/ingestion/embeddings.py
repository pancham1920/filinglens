"""Generate normalized BGE embeddings locally."""

import argparse
import math
from functools import lru_cache
from pathlib import Path

from app.db.connection import PROJECT_ROOT


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
EMBEDDING_DIMENSIONS = 384
MAX_INPUT_TOKENS = 512
PREPROCESSING_VERSION = "passage-v1"

QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)

MODEL_CACHE = PROJECT_ROOT / "data/models"
DEFAULT_CHUNKS = (
    PROJECT_ROOT
    / "data/chunks/0000320193-24-000123.jsonl"
)


class EmbeddingError(RuntimeError):
    pass


@lru_cache(maxsize=2)
def get_model(*, offline: bool = False):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise EmbeddingError(
            "Sentence Transformers is missing. Run uv sync."
        ) from None

    try:
        model = SentenceTransformer(
            EMBEDDING_MODEL,
            revision=MODEL_REVISION,
            cache_folder=str(MODEL_CACHE),
            device="cpu",
            local_files_only=offline,
            trust_remote_code=False,
        )
    except (OSError, RuntimeError, ValueError):
        raise EmbeddingError(
            "Could not load BGE. Offline mode requires "
            "the model to be downloaded first."
        ) from None

    dimension_method = getattr(
        model,
        "get_embedding_dimension",
        None,
    )

    if dimension_method is None:
        dimension_method = model.get_sentence_embedding_dimension

    if dimension_method() != EMBEDDING_DIMENSIONS:
        raise EmbeddingError("Unexpected model dimensions")

    return model


def token_count(text: str, *, offline: bool = False) -> int:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Embedding input must be nonempty text")

    tokenizer = get_model(offline=offline).tokenizer

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
        )
    )


def validate_inputs(
    texts: list[str],
    *,
    offline: bool = False,
) -> None:
    # Validate strings before loading the model.
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding input must be nonempty text")

    if not texts:
        return

    model = get_model(offline=offline)
    limit = min(MAX_INPUT_TOKENS, model.max_seq_length)

    for index, text in enumerate(texts):
        count = token_count(text, offline=offline)

        if count > limit:
            raise EmbeddingError(
                f"Input {index} has {count} tokens; limit is {limit}. "
                "Split it first; no text was truncated."
            )


def validate_vector(vector: object) -> list[float]:
    if (
        not isinstance(vector, list)
        or len(vector) != EMBEDDING_DIMENSIONS
        or any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            for value in vector
        )
        or not any(value != 0 for value in vector)
    ):
        raise EmbeddingError(
            "Expected a finite, nonzero 384-dimensional vector"
        )

    return [float(value) for value in vector]


def embed_text(
    text: str,
    *,
    is_query: bool = False,
    offline: bool = False,
) -> list[float]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Embedding input must be nonempty text")

    input_text = (
        QUERY_INSTRUCTION + text
        if is_query
        else text
    )

    validate_inputs([input_text], offline=offline)

    vector = get_model(offline=offline).encode(
        input_text,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).tolist()

    return validate_vector(vector)


def embed_texts(
    texts: list[str],
    *,
    batch_size: int = 16,
    offline: bool = False,
) -> list[list[float]]:
    """Return one vector per input, in the same order."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    # Check every input before encoding anything.
    validate_inputs(texts, offline=offline)

    if not texts:
        return []

    vectors = get_model(offline=offline).encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).tolist()

    if len(vectors) != len(texts):
        raise EmbeddingError(
            "Embedding count does not match input count"
        )

    return [validate_vector(vector) for vector in vectors]


def audit_chunks(path: Path, *, offline: bool = False) -> int:
    from app.ingestion.chunker import Chunk

    chunks = [
        Chunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]

    if not chunks:
        raise ValueError("Chunk export is empty")

    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("Chunk export contains duplicate IDs")

    model = get_model(offline=offline)
    limit = min(MAX_INPUT_TOKENS, model.max_seq_length)

    oversized = 0
    maximum = 0

    for chunk in chunks:
        count = token_count(chunk.text, offline=offline)
        maximum = max(maximum, count)

        if count > limit:
            oversized += 1
            print(
                f"SPLIT REQUIRED: {chunk.section} / "
                f"{chunk.position} "
                f"({chunk.content_type}, {count} tokens)"
            )

    print(f"Checked chunks: {len(chunks)}")
    print(f"Token limit: {limit}; maximum observed: {maximum}")
    print(f"Chunks requiring splitting: {oversized}")

    return 1 if oversized else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--check-chunks",
        type=Path,
        nargs="?",
        const=DEFAULT_CHUNKS,
    )
    args = parser.parse_args()

    print(f"Model: {EMBEDDING_MODEL}")

    try:
        if args.check_chunks is not None:
            return audit_chunks(
                args.check_chunks,
                offline=args.offline,
            )

        text = "This is a test sentence for FilingLens."
        vector = embed_text(text, offline=args.offline)

        print(f"Dimensions: {len(vector)}")
        print(f"First five values: {vector[:5]}")
        print("PASS: generated a valid embedding locally")
        return 0

    except (EmbeddingError, ValueError, OSError) as error:
        print(f"FAIL: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())