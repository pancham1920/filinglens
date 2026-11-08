"""Validated, content-addressed cache for passage embeddings."""

import hashlib
import json
from pathlib import Path

from app.db.connection import PROJECT_ROOT
from app.ingestion.downloader import save_atomically
from app.ingestion.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    MODEL_REVISION,
    PREPROCESSING_VERSION,
    EmbeddingError,
    validate_vector,
)


CACHE_DIR = PROJECT_ROOT / "data/embeddings"


def embedding_profile() -> dict:
    return {
        "model": EMBEDDING_MODEL,
        "revision": MODEL_REVISION,
        "dimensions": EMBEDDING_DIMENSIONS,
        "normalize_embeddings": True,
        "preprocessing_version": PREPROCESSING_VERSION,
        "input_type": "passage",
    }


def make_cache_key(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Cache input must be nonempty text")

    identity = {
        "profile": embedding_profile(),
        "text": text,
    }

    payload = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_cached_vector(
    text: str,
    *,
    cache_dir: Path = CACHE_DIR,
) -> list[float] | None:
    key = make_cache_key(text)
    path = cache_dir / f"{key}.json"

    if not path.exists():
        return None

    try:
        record = json.loads(path.read_text(encoding="utf-8"))

        if (
            record["cache_key"] != key
            or record["profile"] != embedding_profile()
        ):
            raise EmbeddingError("Cache identity mismatch")

        return validate_vector(record["embedding"])

    except (ValueError, KeyError, TypeError, EmbeddingError):
        raise EmbeddingError(
            f"Invalid cache entry: {path.name}. "
            "Remove that entry before regenerating it."
        ) from None


def save_cached_vector(
    text: str,
    vector: list[float],
    *,
    cache_dir: Path = CACHE_DIR,
) -> None:
    key = make_cache_key(text)

    record = {
        "cache_key": key,
        "profile": embedding_profile(),
        "embedding": validate_vector(vector),
    }

    payload = json.dumps(
        record,
        allow_nan=False,
        separators=(",", ":"),
    )

    save_atomically(
        cache_dir / f"{key}.json",
        payload.encode("utf-8"),
    )