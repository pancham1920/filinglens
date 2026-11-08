"""Prepare, embed, cache, and export the Apple sample."""

import argparse
import json
from pathlib import Path

from app.db.connection import PROJECT_ROOT
from app.ingestion.chunker import Chunk
from app.ingestion.downloader import save_atomically
from app.ingestion.embedding_cache import (
    CACHE_DIR,
    embedding_profile,
    load_cached_vector,
    make_cache_key,
    save_cached_vector,
)
from app.ingestion.embeddings import (
    embed_texts,
    validate_inputs,
    validate_vector,
)
from app.ingestion.export_chunks import ACCESSION
from app.ingestion.prepare_chunks import prepare_file


OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/embedded_chunks"
    / f"{ACCESSION}.jsonl"
)


def embed_chunks(
    chunks: list[Chunk],
    *,
    batch_size: int = 16,
    cache_dir: Path = CACHE_DIR,
) -> list[dict]:
    if not chunks:
        raise ValueError("No chunks to embed")

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk IDs")

    texts = [chunk.text for chunk in chunks]

    # Reject invalid inputs even when a cache entry exists.
    validate_inputs(texts, offline=True)

    vectors = {}
    missing = {}
    cache_hits = 0

    for text in texts:
        key = make_cache_key(text)

        if key in vectors:
            cache_hits += 1
            continue

        if key in missing:
            continue

        cached = load_cached_vector(
            text,
            cache_dir=cache_dir,
        )

        if cached is None:
            missing[key] = text
        else:
            vectors[key] = cached
            cache_hits += 1

    items = list(missing.items())

    print(f"Cached chunk references: {cache_hits}")
    print(f"Distinct texts requiring embedding: {len(items)}")

    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        batch_texts = [text for _, text in batch]

        generated = embed_texts(
            batch_texts,
            batch_size=batch_size,
            offline=True,
        )

        if len(generated) != len(batch):
            raise ValueError("Batch result count mismatch")

        for (key, text), vector in zip(
            batch,
            generated,
            strict=True,
        ):
            vector = validate_vector(vector)

            save_cached_vector(
                text,
                vector,
                cache_dir=cache_dir,
            )

            vectors[key] = vector

        print(
            f"Embedded and cached: "
            f"{min(start + batch_size, len(items))}/{len(items)}"
        )

    # Assemble in input order, regardless of where vectors came from.
    records = []

    for chunk in chunks:
        key = make_cache_key(chunk.text)

        records.append({
            "chunk": chunk.model_dump(mode="json"),
            "embedding_profile": embedding_profile(),
            "cache_key": key,
            "embedding": validate_vector(vectors[key]),
        })

    print(f"Validated chunk/vector pairs: {len(records)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    chunks = prepare_file()

    records = embed_chunks(
        chunks,
        batch_size=args.batch_size,
    )

    payload = "".join(
        json.dumps(record, allow_nan=False) + "\n"
        for record in records
    )

    save_atomically(
        OUTPUT_PATH,
        payload.encode("utf-8"),
    )

    restored = [
        json.loads(line)
        for line in OUTPUT_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    if restored != records:
        raise ValueError("Embedded export read-back failed")

    print(f"Saved: {OUTPUT_PATH}")
    print("PASS: embedded export matches chunk/vector pairs")


if __name__ == "__main__":
    main()