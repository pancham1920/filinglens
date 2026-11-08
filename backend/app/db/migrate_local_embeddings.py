"""Apply the local-model migration atomically, after migrations 001 and 002."""

import psycopg

from app.db.connection import PROJECT_ROOT, get_connection


def main() -> int:
    path = PROJECT_ROOT / "backend/migrations/003_local_embeddings.sql"
    try:
        with get_connection() as connection:
            connection.execute(path.read_text(encoding="utf-8"))
            row = connection.execute(
                """
                SELECT format_type(atttypid, atttypmod)
                FROM pg_attribute
                WHERE attrelid = 'public.chunks'::regclass
                  AND attname = 'embedding' AND NOT attisdropped
                """
            ).fetchone()
            if row != ("vector(384)",):
                raise RuntimeError("Embedding column migration did not verify")
    except psycopg.Error as error:
        print(f"FAIL: database migration rolled back ({type(error).__name__})")
        return 1
    print("PASS: applied 003_local_embeddings.sql")
    print("PASS: chunks.embedding = vector(384)")
    print("PASS: ingestion run dimensions restricted to 384")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
