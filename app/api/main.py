"""
FastAPI application — app/api/main.py

REST API for NeuroGraph. Exposes the pipeline as HTTP endpoints
with structured JSON responses, error handling, and request validation.

Endpoints:
    POST /query      — run the full pipeline on a question
    POST /index      — index PDFs from data/raw/
    GET  /health     — health check (LLM, vector store status)
    GET  /stats      — system statistics (chunks, token usage)

Usage:
    uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)


# Request/response models

class QueryRequest(BaseModel):
    """Request body for /query endpoint."""
    query: str = Field(..., min_length=1, max_length=2000, description="The question to answer")
    strategy: str | None = Field(None, description="Override retrieval strategy: vector, keyword, hybrid")
    top_k: int | None = Field(None, ge=1, le=32, description="Number of chunks to retrieve")


class QueryResponse(BaseModel):
    """Response body for /query endpoint."""
    answer: str
    sources: list[dict]
    confidence: float
    timing: dict
    errors: list[str]
    plan: dict | None = None
    verdict: dict | None = None


class IndexRequest(BaseModel):
    """Request body for /index endpoint."""
    pdf_dir: str | None = Field(None, description="Override PDF directory path")
    collection_name: str = Field("neurograph", description="ChromaDB collection name")


class IndexResponse(BaseModel):
    """Response body for /index endpoint."""
    chunks_indexed: int
    pdf_count: int
    collection: str
    time_ms: float


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    status: str
    llm_provider: str
    llm_available: bool
    vector_store_count: int
    embedding_provider: str


# Application lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("NeuroGraph API starting up")
    logger.info("LLM provider: %s", settings.llm_provider)
    logger.info("Embedding model: %s", settings.embedding_model)
    yield
    logger.info("NeuroGraph API shutting down")


# Application

app = FastAPI(
    title="NeuroGraph",
    description="Multi-agent RAG research assistant for neuroscience",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Run the full pipeline on a question.

    Returns a cited answer with sources, confidence score,
    and optional trace data for debugging.
    """
    from app.pipeline import Pipeline

    try:
        pipeline = Pipeline()
        result = pipeline.run(
            query=request.query,
            strategy=request.strategy,
            top_k=request.top_k,
        )

        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            confidence=result.confidence,
            timing=result.timing,
            errors=result.errors,
            plan=result.plan.to_dict() if result.plan else None,
            verdict=result.verdict.to_dict() if result.verdict else None,
        )

    except Exception as e:
        logger.error("Query failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index", response_model=IndexResponse)
async def index_pdfs(request: IndexRequest):
    """
    Index all PDFs in data/raw/ into the vector store.

    Run this after adding new PDFs to build or update
    the search index.
    """
    from app.rag.loader import load_all_pdfs
    from app.rag.embedder import build_index

    start = time.monotonic()

    try:
        pdf_dir = Path(request.pdf_dir) if request.pdf_dir else settings.data_raw

        if not pdf_dir.exists():
            raise HTTPException(
                status_code=400,
                detail=f"PDF directory not found: {pdf_dir}",
            )

        pdf_count = len(list(pdf_dir.glob("*.pdf")))
        if pdf_count == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No PDF files found in {pdf_dir}",
            )

        chunks = load_all_pdfs()
        store = build_index(chunks)

        elapsed = (time.monotonic() - start) * 1000

        return IndexResponse(
            chunks_indexed=store.count,
            pdf_count=pdf_count,
            collection=request.collection_name,
            time_ms=round(elapsed, 1),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Indexing failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check system health: LLM availability and vector store status.
    """
    from app.rag.embedder import get_store, get_embedder

    try:
        store = get_store()
        embedder = get_embedder()

        llm_available = False
        try:
            from app.core.llm import get_client
            llm_available = get_client().is_available()
        except Exception:
            pass

        return HealthResponse(
            status="healthy" if llm_available else "degraded",
            llm_provider=settings.llm_provider,
            llm_available=llm_available,
            vector_store_count=store.count,
            embedding_provider=embedder.provider_name,
        )

    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            llm_provider=settings.llm_provider,
            llm_available=False,
            vector_store_count=0,
            embedding_provider="unavailable",
        )


@app.get("/stats")
async def stats():
    """System statistics: chunk count, token usage, config."""
    from app.core.llm import get_client

    try:
        client = get_client()
        token_usage = client.token_usage
    except Exception:
        token_usage = {}

    try:
        from app.rag.embedder import get_store
        store = get_store()
        chunk_count = store.count
    except Exception:
        chunk_count = 0

    return {
        "chunks_indexed": chunk_count,
        "token_usage": token_usage,
        "config": {
            "llm_provider": settings.llm_provider,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
        },
    }