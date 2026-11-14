"""Vector, keyword, and hybrid search over indexed filing chunks."""

import argparse
import json
import re

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.db.connection import get_connection
from app.ingestion.embedding_cache import embedding_profile
from app.ingestion.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EmbeddingError,
    embed_text,
)


def build_filters(
    ticker=None,
    fiscal_year=None,
    section=None,
    content_type=None,
):
    conditions = []
    values = []

    if ticker is not None:
        ticker = ticker.strip().upper()

        if not ticker:
            raise ValueError("Ticker cannot be empty")

        conditions.append("upper(co.ticker) = %s")
        values.append(ticker)

    if fiscal_year is not None:
        if not 1900 <= fiscal_year <= 2100:
            raise ValueError("Fiscal year must be between 1900 and 2100")

        conditions.append("f.fiscal_year = %s")
        values.append(fiscal_year)

    if section is not None:
        section = section.strip()

        if not section:
            raise ValueError("Section cannot be empty")

        conditions.append("lower(c.section) = lower(%s)")
        values.append(section)

    if content_type is not None:
        if content_type not in ("prose", "table"):
            raise ValueError("Content type must be prose or table")

        conditions.append("c.content_type = %s")
        values.append(content_type)

    sql = "".join(f" AND {condition}" for condition in conditions)
    return sql, values


def vector_search(
    query: str,
    limit: int = 5,
    *,
    ticker=None,
    fiscal_year=None,
    section=None,
    content_type=None,
) -> list[dict]:
    if not query.strip():
        raise ValueError("Enter a nonempty search query")

    if not 1 <= limit <= 20:
        raise ValueError("Limit must be between 1 and 20")

    filter_sql, filter_values = build_filters(
        ticker,
        fiscal_year,
        section,
        content_type,
    )

    vector = embed_text(
        query.strip(),
        is_query=True,
        offline=True,
    )

    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                f"""
                SELECT
                    c.chunk_id::text AS chunk_id,
                    co.ticker,
                    co.name AS company,
                    f.fiscal_year,
                    c.section,
                    c.content_type,
                    c.content AS text,
                    c.source_locator AS citation_url,
                    c.embedding <=> %s::vector(384) AS distance
                FROM chunks c
                JOIN ingestion_runs r
                    ON r.id = c.ingestion_run_id
                JOIN filings f
                    ON f.accession_number = r.filing_accession
                JOIN companies co
                    ON co.cik = f.company_cik
                WHERE r.active = TRUE
                  AND r.status = 'completed'
                  AND r.embedding_model = %s
                  AND r.embedding_dimensions = %s
                  AND c.embedding IS NOT NULL
                  AND c.source_metadata->'embedding_profile' = %s
                  {filter_sql}
                ORDER BY distance, c.id
                LIMIT %s
                """,
                (
                    json.dumps(vector),
                    EMBEDDING_MODEL,
                    EMBEDDING_DIMENSIONS,
                    Jsonb(embedding_profile()),
                    *filter_values,
                    limit,
                ),
            )

            results = cursor.fetchall()

    for result in results:
        result["score"] = 1 - result.pop("distance")

    return results


def keyword_search(
    query: str,
    limit: int = 5,
    *,
    ticker=None,
    fiscal_year=None,
    section=None,
    content_type=None,
) -> list[dict]:
    if not query.strip():
        raise ValueError("Enter a nonempty search query")

    if not 1 <= limit <= 20:
        raise ValueError("Limit must be between 1 and 20")

    filter_sql, filter_values = build_filters(
        ticker,
        fiscal_year,
        section,
        content_type,
    )

    # Tables may contain the year without the word "fiscal".
    keyword_query = re.sub(
        r"\bfiscal(?:\s+year)?\s+(?=\d{4}\b)",
        "",
        query.strip(),
        flags=re.IGNORECASE,
    )

    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                f"""
                WITH search_query AS (
                    SELECT websearch_to_tsquery(
                        'english', %s
                    ) AS terms
                ),
                documents AS (
                    SELECT
                        c.chunk_id::text AS chunk_id,
                        co.ticker,
                        co.name AS company,
                        f.fiscal_year,
                        c.section,
                        c.content_type,
                        c.content AS text,
                        c.source_locator AS citation_url,
                        to_tsvector(
                            'english',
                            concat_ws(
                                ' ',
                                co.name,
                                co.ticker,
                                f.fiscal_year::text,
                                c.section,
                                c.content
                            )
                        ) AS document
                    FROM chunks c
                    JOIN ingestion_runs r
                        ON r.id = c.ingestion_run_id
                    JOIN filings f
                        ON f.accession_number = r.filing_accession
                    JOIN companies co
                        ON co.cik = f.company_cik
                    WHERE r.active = TRUE
                      AND r.status = 'completed'
                      AND r.embedding_model = %s
                      AND r.embedding_dimensions = %s
                      AND c.embedding IS NOT NULL
                      AND c.source_metadata->'embedding_profile' = %s
                      {filter_sql}
                )
                SELECT
                    d.chunk_id,
                    d.ticker,
                    d.company,
                    d.fiscal_year,
                    d.section,
                    d.content_type,
                    d.text,
                    d.citation_url,
                    ts_rank(d.document, q.terms) AS score
                FROM documents d
                CROSS JOIN search_query q
                WHERE d.document @@ q.terms
                ORDER BY score DESC, d.chunk_id
                LIMIT %s
                """,
                (
                    keyword_query,
                    EMBEDDING_MODEL,
                    EMBEDDING_DIMENSIONS,
                    Jsonb(embedding_profile()),
                    *filter_values,
                    limit,
                ),
            )

            return cursor.fetchall()


def hybrid_search(
    query: str,
    limit: int = 5,
    *,
    ticker=None,
    fiscal_year=None,
    section=None,
    content_type=None,
) -> list[dict]:
    if not query.strip():
        raise ValueError("Enter a nonempty search query")

    if not 1 <= limit <= 20:
        raise ValueError("Limit must be between 1 and 20")

    filters = {
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "section": section,
        "content_type": content_type,
    }

    vector_results = vector_search(query, limit=20, **filters)
    keyword_results = keyword_search(query, limit=20, **filters)

    combined = {}

    for rank_field, results in (
        ("vector_rank", vector_results),
        ("keyword_rank", keyword_results),
    ):
        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]

            if chunk_id not in combined:
                combined[chunk_id] = {
                    **result,
                    "score": 0.0,
                    "vector_rank": None,
                    "keyword_rank": None,
                }

            # Give the first few positions more influence.
            combined[chunk_id]["score"] += 1 / (1 + rank)
            combined[chunk_id][rank_field] = rank

    ranked = sorted(
        combined.values(),
        key=lambda result: (-result["score"], result["chunk_id"]),
    )

    return ranked[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--method",
        choices=["vector", "keyword", "hybrid"],
        default="hybrid",
    )
    parser.add_argument("--ticker")
    parser.add_argument("--year", dest="fiscal_year", type=int)
    parser.add_argument("--section")
    parser.add_argument(
        "--type",
        dest="content_type",
        choices=["prose", "table"],
    )
    args = parser.parse_args()

    filters = {
        "ticker": args.ticker,
        "fiscal_year": args.fiscal_year,
        "section": args.section,
        "content_type": args.content_type,
    }

    try:
        if args.method == "vector":
            results = vector_search(
                args.query, args.limit, **filters
            )
        elif args.method == "keyword":
            results = keyword_search(
                args.query, args.limit, **filters
            )
        else:
            results = hybrid_search(
                args.query, args.limit, **filters
            )

    except (ValueError, EmbeddingError) as error:
        print(f"Search failed: {error}")
        return 1
    except psycopg.Error as error:
        print(f"Database search failed: {type(error).__name__}")
        return 1

    if not results:
        print("No matching indexed chunks.")
        return 0

    for position, result in enumerate(results, start=1):
        print(
            f"\n{position}. {result['ticker']} "
            f"{result['fiscal_year']} — {result['section']}"
        )
        print(
            f"Type: {result['content_type']} | "
            f"Score: {result['score']:.4f}"
        )

        if args.method == "hybrid":
            print(
                f"Vector rank: {result['vector_rank']} | "
                f"Keyword rank: {result['keyword_rank']}"
            )

        print(result["text"][:600])
        print(f"Source: {result['citation_url']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())