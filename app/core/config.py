import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    """All NeuroGraph settings in one place."""
    project_root: Path = Path(__file__).resolve().parent.parent.parent

    @property
    def data_raw(self) -> Path:
        """Raw data This is where you drop your PDF's and notes."""
        return self.project_root / 'data' / 'raw'

    @property
    def data_processed(self) -> Path:
        """Where chunked documents get saved as JSON."""
        return self.project_root / "data" / "processed"

    @property
    def chroma_path(self) -> Path:
        """Where ChromaDB stores its vector index on disk."""
        return self.project_root / "data" / "index" / "chroma"


    # Chunking
    # chunk_size: how many characters per chunk (500 is good for papers)
    # chunk_overlap: shared characters between consecutive chunks
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrival
    # how meny top chunks to return from the vector search
    top_k: int = 15


    # "ollama" = local, free, needs ~8GB RAM
    # "groq"   = cloud, free tier, 14400 requests/day
    # LLM provider
    llm_provider: str = "ollama"

    # ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    embedding_model: str = "nomic-embed-text"

    # Groq settings
    groq_model: str = "llama-3.1-8b-instant"

    @property
    def groq_api_key(self) -> str:
        """Read Groq API key from environment. Never hardcoded."""
        key = os.environ.get("GROQ_API_KEY", "")
        if self.llm_provider == "groq" and not key:
            raise ValueError(
                "GROQ_API_KEY not set. Get a free key at console.groq.com"
            )
        return key


# Singleton
# One instance, shared by every module that imports it.
# Usage: from app.core.config import settings
settings = Settings()

