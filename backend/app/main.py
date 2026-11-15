from typing import Literal

import psycopg
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.answer import answer_question

from app.ingestion.embeddings import EmbeddingError
from app.retrieval import (
    hybrid_search,
    keyword_search,
    vector_search,
)


app = FastAPI(title="FilingLens API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_methods=["GET","POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, max_length=1000),
    method: Literal["vector", "keyword", "hybrid"] = "hybrid",
    limit: int = Query(5, ge=1, le=20),
    ticker: str | None = None,
    year: int | None = Query(None, ge=1900, le=2100),
    section: str | None = None,
    content_type: Literal["prose", "table"] | None = Query(
        None, alias="type"
    ),
):
    search_function = {
        "vector": vector_search,
        "keyword": keyword_search,
        "hybrid": hybrid_search,
    }[method]

    try:
        results = search_function(
            q,
            limit,
            ticker=ticker,
            fiscal_year=year,
            section=section,
            content_type=content_type,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from None
    except EmbeddingError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from None
    except psycopg.Error:
        raise HTTPException(
            status_code=503,
            detail="Database search is currently unavailable",
        ) from None

    return {
        "query": q.strip(),
        "method": method,
        "count": len(results),
        "results": results,
    }

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    ticker: str | None = None
    year: int | None = Field(default=None, ge=1900, le=2100)
    section: str | None = None
    content_type: Literal["prose", "table"] | None = Field(
        default=None,
        alias="type",
    )


@app.post("/ask")
def ask(request: AskRequest):
    try:
        result = answer_question(
            request.question,
            ticker=request.ticker,
            fiscal_year=request.year,
            section=request.section,
            content_type=request.content_type,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from None
    except psycopg.Error:
        raise HTTPException(
            status_code=503,
            detail="Database search is currently unavailable",
        ) from None
    except (EmbeddingError, RuntimeError) as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from None

    return {
        "question": request.question.strip(),
        **result,
    }