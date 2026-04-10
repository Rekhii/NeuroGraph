"""
Embedding and vector storage — app/rag/embedder.py

Converts text chunks into vector embeddings and stores them
in ChromaDB for similarity search.

Two embedding providers supported:
    - Ollama (local): nomic-embed-text via localhost:11434
    - HuggingFace (fallback): all-MiniLM-L6-v2 via sentence-transformers

Architecture:
    chunks from loader.py
      -> EmbeddingProvider (abstracts the model)
      -> VectorStore (wraps ChromaDB)
      -> persisted to disk at data/index/chroma/

Design decisions:
    - Batched upserts to avoid memory spikes on large corpora
    - Idempotent: re-running on the same chunks skips duplicates
    - Provider pattern so swapping embedding models is one-line
    - Collection name is configurable for multi-corpus support
    - Embedding validation catches silent model failures
    - Retry logic handles transient Ollama server issues
    - Content hashing detects duplicate chunks across re-indexing runs
    - Similarity scores (0-1) instead of raw cosine distance
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Protocol, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.rag.loader import ChunkData

logger = logging.getLogger(__name__)


# Constants

MAX_CHUNK_CHARS = 8000      # chunks longer than this get truncated before embedding
MIN_CHUNK_CHARS = 10        # chunks shorter than this get skipped
RETRY_ATTEMPTS = 3          # how many times to retry a failed embed call
RETRY_DELAY = 2.0           # seconds between retries


# Embedding validation

def _validate_embeddings(
    texts: list[str],
    embeddings: list[list[float]],
) -> None:
    """
    Check that embeddings are well-formed before storing them.

    Catches three failure modes:
    1. Model returned fewer/more vectors than input texts
    2. Vectors have inconsistent dimensions across the batch
    3. Vectors are empty (model returned placeholder)
    """
    if len(embeddings) != len(texts):
        raise ValueError(
            f"Embedding count mismatch: got {len(embeddings)} "
            f"embeddings for {len(texts)} texts"
        )

    if not embeddings:
        return

    expected_dim = len(embeddings[0])
    if expected_dim == 0:
        raise ValueError("Embedding dimension is 0, model returned empty vectors")

    for i, vec in enumerate(embeddings):
        if len(vec) != expected_dim:
            raise ValueError(
                f"Dimension mismatch at index {i}: "
                f"expected {expected_dim}, got {len(vec)}"
            )


# Content hashing

def _content_hash(text: str) -> str:
    """
    SHA-256 hex digest of chunk text, truncated to 16 chars.

    Used to detect duplicate content across different chunking runs.
    If chunk boundaries shift slightly but content is the same,
    the hash catches it. Stored in metadata for dedup queries.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# Chunk sanitization

def _sanitize_chunks(chunks: Sequence[ChunkData]) -> list[ChunkData]:
    """
    Filter and truncate chunks before embedding.

    PDFs can produce irregular outputs even after splitting:
    - Extremely long strings from malformed pages
    - Near-empty chunks from whitespace-only content
    - Noisy text with encoding artifacts

    This step normalizes everything before it hits the model.
    """
    clean: list[ChunkData] = []

    for chunk in chunks:
        text = chunk["text"].strip()

        # Skip chunks that are too short to be meaningful
        if len(text) < MIN_CHUNK_CHARS:
            logger.debug("Skipping short chunk: %s (%d chars)", chunk["chunk_id"], len(text))
            continue

        # Truncate chunks that are abnormally long
        if len(text) > MAX_CHUNK_CHARS:
            logger.warning(
                "Truncating oversized chunk: %s (%d -> %d chars)",
                chunk["chunk_id"], len(text), MAX_CHUNK_CHARS,
            )
            text = text[:MAX_CHUNK_CHARS]

        clean.append({
            "text": text,
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
        })

    skipped = len(chunks) - len(clean)
    if skipped:
        logger.info("Sanitization: kept %d, skipped %d chunks", len(clean), skipped)

    return clean


# Embedding providers

class EmbeddingProvider(Protocol):
    """Interface that any embedding backend must satisfy."""
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def provider_name(self) -> str: ...


class OllamaEmbedder:
    """
    Generate embeddings via Ollama's local API.

    Ollama exposes /api/embed which accepts a list of texts
    and returns a list of float vectors. No API key needed.
    Requires: ollama pull nomic-embed-text

    Includes retry logic for transient server failures.
    Local model servers are not perfectly stable under load,
    and a single timeout should not kill a large indexing job.
    """

    def __init__(
        self,
        model: str = settings.embedding_model,
        base_url: str = settings.ollama_base_url,
        retries: int = RETRY_ATTEMPTS,
        retry_delay: float = RETRY_DELAY,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self.retry_delay = retry_delay

    @property
    def provider_name(self) -> str:
        return f"ollama/{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        last_error: Exception | None = None

        for attempt in range(1, self.retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": texts},
                    timeout=120.0,
                )
                response.raise_for_status()
                embeddings = response.json()["embeddings"]
                _validate_embeddings(texts, embeddings)
                return embeddings

            except Exception as e:
                last_error = e
                if attempt < self.retries:
                    logger.warning(
                        "Ollama embed attempt %d/%d failed: %s. Retrying in %.1fs",
                        attempt, self.retries, e, self.retry_delay,
                    )
                    time.sleep(self.retry_delay)

        raise RuntimeError(
            f"Ollama embedding failed after {self.retries} attempts"
        ) from last_error


class HuggingFaceEmbedder:
    """
    Fallback embedder using sentence-transformers.

    Runs entirely on CPU. No API key, no server, no GPU needed.
    Slower than Ollama but works everywhere.
    Requires: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return f"huggingface/{self._model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, show_progress_bar=False)
        embeddings = vectors.tolist()
        _validate_embeddings(texts, embeddings)
        return embeddings


def get_embedder() -> EmbeddingProvider:
    """
    Factory that returns the right embedder based on config.

    Tries Ollama first (fast, local). Falls back to HuggingFace
    if Ollama is not reachable. Logs the active provider explicitly
    so you always know which model is generating your vectors.
    """
    if settings.llm_provider == "ollama":
        try:
            import httpx
            resp = httpx.get(
                f"{settings.ollama_base_url}/api/tags",
                timeout=5.0,
            )
            resp.raise_for_status()
            embedder = OllamaEmbedder()
            logger.info("Embedding provider active: %s", embedder.provider_name)
            return embedder
        except Exception as e:
            logger.warning(
                "Ollama not reachable (%s), falling back to HuggingFace", e
            )

    embedder = HuggingFaceEmbedder()
    logger.info("Embedding provider active: %s", embedder.provider_name)
    return embedder


# Vector store

class VectorStore:
    """
    Wraps ChromaDB with batched upserts and metadata handling.

    ChromaDB stores three things per document:
        1. The embedding vector (for similarity search)
        2. The raw text (returned with results)
        3. Metadata dict (source, page, content_hash, text_length)

    Key behaviors:
        - Persists to disk so you embed once, search many times
        - Uses upsert (not add) so re-running is safe
        - Batches large inserts to cap memory usage
        - Validates embeddings before storage
        - Returns similarity scores (0-1) not raw distance
    """

    DEFAULT_COLLECTION = "neurograph"
    BATCH_SIZE = 100

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        persist_dir: str | None = None,
        embedder: EmbeddingProvider | None = None,
    ):
        self._persist_dir = persist_dir or str(settings.chroma_path)

        # Ensure persist directory exists before ChromaDB tries to use it.
        # New environments or fresh deployments may not have it.
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        self._embedder = embedder or get_embedder()

        # ChromaDB client with disk persistence
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # get_or_create is idempotent: first call creates,
        # subsequent calls return the existing collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        """Number of documents currently in the collection."""
        return self._collection.count()

    @property
    def collection_name(self) -> str:
        return self._collection.name

    @property
    def active_provider(self) -> str:
        """Which embedding model is currently in use."""
        if hasattr(self._embedder, "provider_name"):
            return self._embedder.provider_name
        return "unknown"

    def add_chunks(self, chunks: Sequence[ChunkData]) -> int:
        """
        Sanitize, embed, and store chunks in batches.

        Pipeline per batch:
        1. Sanitize (skip short, truncate long)
        2. Embed (with retry on Ollama)
        3. Validate (check dimensions match)
        4. Upsert (idempotent write to ChromaDB)

        Returns the number of chunks successfully stored.
        """
        if not chunks:
            return 0

        # Sanitize before embedding
        clean_chunks = _sanitize_chunks(chunks)
        if not clean_chunks:
            return 0

        total = 0

        for start in range(0, len(clean_chunks), self.BATCH_SIZE):
            batch = clean_chunks[start : start + self.BATCH_SIZE]

            ids = [c["chunk_id"] for c in batch]
            texts = [c["text"] for c in batch]

            # Rich metadata for debugging, ranking, and dedup
            metadatas = [
                {
                    "source": c["source"],
                    "page": c["page"],
                    "chunk_id": c["chunk_id"],
                    "text_length": len(c["text"]),
                    "content_hash": _content_hash(c["text"]),
                }
                for c in batch
            ]

            # Generate and validate embeddings
            embeddings = self._embedder.embed(texts)

            # Upsert: insert new, overwrite existing (by chunk_id)
            self._collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            total += len(batch)
            logger.debug("Upserted batch %d-%d", start, start + len(batch))

        logger.info(
            "Stored %d chunks in '%s' (provider: %s)",
            total, self.collection_name, self.active_provider,
        )
        return total

    def query(
        self,
        query_text: str,
        top_k: int = settings.top_k,
    ) -> list[dict]:
        """
        Search for chunks similar to the query text.

        Returns a list of result dicts sorted by relevance:
            - text: the chunk content
            - source: which PDF
            - page: which page
            - chunk_id: unique identifier
            - score: similarity score 0-1 (higher = more similar)
            - distance: raw cosine distance (lower = more similar)
        """
        query_embedding = self._embedder.embed([query_text])[0]

        n = min(top_k, self.count) if self.count > 0 else top_k

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns nested lists (supports batch queries).
        # We only query one text at a time, so take index [0].
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []
        ids = results["ids"][0] if results["ids"] else []

        return [
            {
                "text": doc,
                "source": meta.get("source", ""),
                "page": meta.get("page", 0),
                "chunk_id": cid,
                "score": round(1.0 - dist, 4),
                "distance": dist,
            }
            for doc, meta, dist, cid in zip(documents, metadatas, distances, ids)
        ]

    def get_all_documents(self) -> list[dict]:
        """
        Return every document in the collection with metadata.

        Used by BM25Index to build a keyword search index from
        the same corpus that vector search uses. Provides a
        clean public interface so no external code needs to
        touch the internal _collection object.

        Returns:
            List of dicts with keys: text, source, page, chunk_id
        """
        if self.count == 0:
            return []

        raw = self._collection.get(
            include=["documents", "metadatas"],
        )

        documents = []
        for doc_id, text, meta in zip(
                raw["ids"],
                raw["documents"],
                raw["metadatas"],
        ):
            documents.append({
                "text": text,
                "source": meta.get("source", ""),
                "page": meta.get("page", 0),
                "chunk_id": doc_id,
            })

        return documents

    def get_indexed_ids(self) -> set[str]:
        """
        Return all chunk IDs currently stored in the collection.

        Used for incremental indexing: before embedding new chunks,
        check which IDs already exist and skip them. This avoids
        re-embedding the entire corpus when a single new PDF is added.

        Returns:
            Set of chunk_id strings
        """
        if self.count == 0:
            return set()

        raw = self._collection.get(include=[])
        return set(raw["ids"])

    def delete_collection(self) -> None:
        """Drop the entire collection. Use for reindexing."""
        self._client.delete_collection(self.collection_name)
        logger.info("Deleted collection '%s'", self.collection_name)

    def reset(self) -> None:
        """Delete and recreate the collection. Clean slate."""
        self.delete_collection()
        self._collection = self._client.get_or_create_collection(
            name=self.DEFAULT_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )


# Convenience functions

def build_index(chunks: Sequence[ChunkData]) -> VectorStore:
    """
    One-call function: create a VectorStore and index all chunks.

    Usage:
        from app.rag.loader import load_all_pdfs
        from app.rag.embedder import build_index

        chunks = load_all_pdfs()
        store = build_index(chunks)
        results = store.query("What is active inference?")
    """
    store = VectorStore()
    store.add_chunks(chunks)
    return store


def get_store(collection_name: str = VectorStore.DEFAULT_COLLECTION) -> VectorStore:
    """
    Connect to an existing VectorStore without adding new chunks.

    Use this in the retriever and agents when the index
    already exists and you just need to search it.
    """
    return VectorStore(collection_name=collection_name)