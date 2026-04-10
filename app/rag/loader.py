"""
Document loader — app/rag/loader.py

This is the entry point for all documents in NeuroGraph.
PDFs go in, structured chunks come out.

Pipeline:
    PDF file
      -> extract_text_from_pdf()  (page-level text with metadata)
      -> chunk_pages()            (smaller pieces for vector search)
      -> list[ChunkData]          (ready for embedding)

Why chunk at all?
    LLMs have limited context windows. Vector search works better
    on focused passages than on entire pages. A 500-char chunk about
    "active inference and free energy" will match a query about that
    topic much more precisely than a 3000-char page that also talks
    about five other things.

Usage:
    from app.rag.loader import load_pdf, load_all_pdfs

    chunks = load_pdf("data/raw/baars_gwt.pdf")     # one file
    chunks = load_all_pdfs()                          # everything in data/raw/
"""

from pathlib import Path

# TypedDict lets us define the exact shape of our dictionaries.
# Unlike a plain dict, the editor knows what keys exist and what
# types their values are. Catches typos at dev time, not runtime.
from typing import TypedDict

# RecursiveCharacterTextSplitter tries to split on the best
# boundary it can find. Priority order:
#   1. paragraph breaks ("\n\n")
#   2. single newlines ("\n")
#   3. sentence ends (". ")
#   4. spaces (" ")
#   5. individual characters ("")
# This keeps paragraphs and sentences intact whenever possible.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# pypdf reads PDF files and extracts text page by page.
# Note: this is "pypdf" (lowercase), not the old "PyPDF2" which
# is deprecated. Same authors, new name.
from pypdf import PdfReader

# All settings come from one place. chunk_size, chunk_overlap,
# data paths — nothing is hardcoded in this file.
from app.core.config import settings


# Data shapes
# These TypedDicts define what our functions return.
# Every function downstream knows exactly what keys to expect.

class PageData(TypedDict):
    """One page of extracted PDF text."""
    text: str       # the cleaned text content
    page: int       # 1-indexed page number (humans count from 1)
    source: str     # filename like "baars_gwt.pdf"


class ChunkData(TypedDict):
    text: str
    source: str
    page: int
    chunk_id: str
    has_math: bool      # NEW: whether chunk contains math notation


# Helpers

def _build_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create the text splitter from config settings.

    Isolated in its own function so that:
    1. chunk_pages() stays clean (no setup clutter)
    2. Swapping to a different splitter later (semantic chunking,
       sentence-based, etc.) means changing one function
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,       # 500 chars by default
        chunk_overlap=settings.chunk_overlap,  # 50 chars overlap
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# Core functions

def extract_text_from_pdf(pdf_path: Path) -> list[PageData]:
    """
    Read a PDF and return a list of page dictionaries.

    Extracts text page by page with math-aware cleaning:
    - Normal prose lines: collapse whitespace into single spaces
    - Math lines (containing operators, Greek letters, symbols):
      preserve structure to keep equation formatting intact
    """

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {pdf_path}") from e

    pages: list[PageData] = []
    source = pdf_path.name

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if not text:
            continue

        # Math-aware cleaning: preserve equation lines,
        # collapse whitespace in normal prose lines
        lines = text.strip().split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            has_math = any(c in stripped for c in "=∑∫∂∇±×÷≈≠≤≥∈∀∃∞αβγδεζηθλμνξπρσφψω")
            is_short_formula = len(stripped) < 80 and any(c in stripped for c in "()[]{}=<>^_")
            if has_math or is_short_formula:
                cleaned_lines.append(stripped)
            else:
                cleaned_lines.append(" ".join(stripped.split()))

        cleaned_text = "\n".join(cleaned_lines)

        if not cleaned_text:
            continue

        pages.append(
            {
                "text": cleaned_text,
                "page": page_number,
                "source": source,
            }
        )

    return pages


def chunk_pages(pages: list[PageData]) -> list[ChunkData]:
    """
    Split page-level text into smaller chunks with metadata.

    Why not just use whole pages?
    A page might be 3000 characters covering three different topics.
    When someone asks about "active inference", a 500-char chunk
    focused on that topic scores higher in vector similarity than
    a 3000-char page where it is mentioned once.

    The chunk_id format is: "filename_pN_cM"
    - filename: which PDF
    - pN: which page (1-indexed)
    - cM: which chunk on that page (1-indexed)

    Example: "baars_gwt.pdf_p3_c2" = file baars_gwt.pdf, page 3, chunk 2

    ChromaDB needs unique IDs. This format also makes debugging easy:
    when a chunk looks wrong, you can trace it back to the exact page.
    """
    splitter = _build_splitter()
    all_chunks: list[ChunkData] = []

    for page_data in pages:
        # Split this page's text into chunks
        text_chunks = splitter.split_text(page_data["text"])

        for chunk_index, chunk_text in enumerate(text_chunks, start=1):
            # The splitter can occasionally produce whitespace-only
            # chunks at page boundaries. Strip and skip them.
            cleaned_chunk = chunk_text.strip()
            if not cleaned_chunk:
                continue

            # Build the unique chunk ID
            chunk_id = (
                f"{page_data['source']}"
                f"_p{page_data['page']}"
                f"_c{chunk_index}"
            )

            has_math = any(c in cleaned_chunk for c in "=∑∫∂∇±×÷≈≠≤≥∈∀∃∞αβγδεζηθλμνξπρσφψω")

            all_chunks.append(
                {
                    "text": cleaned_chunk,
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "chunk_id": chunk_id,
                    "has_math": has_math,
                }
            )

    return all_chunks


# Public API

def load_pdf(pdf_path: str | Path) -> list[ChunkData]:
    """
    Full pipeline for one PDF: validate -> extract -> chunk.

    This is the main function for loading a single document.
    It connects extract_text_from_pdf and chunk_pages into
    one clean call.

    Args:
        pdf_path: path to the PDF (string or Path object)

    Returns:
        List of ChunkData dicts ready for embedding.
        Empty list if no text could be extracted.
    """
    path = Path(pdf_path)
    pages = extract_text_from_pdf(path)

    if not pages:
        return []

    return chunk_pages(pages)


def load_all_pdfs() -> list[ChunkData]:
    """
    Load every PDF in data/raw/ and return all chunks combined.

    This is what you run when setting up the knowledge base
    for the first time:
        1. Drop your neuroscience papers into data/raw/
        2. Call load_all_pdfs()
        3. Feed the result into the embedder

    Raises FileNotFoundError if data/raw/ does not exist,
    because a missing directory usually means a path
    misconfiguration, not "zero documents."
    """
    pdf_dir = settings.data_raw

    # Catch path misconfiguration early. A missing directory
    # is different from an empty directory.
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    # sorted() gives deterministic order across runs.
    # is_file() filter skips subdirectories that happen to
    # end in .pdf (unlikely but defensive).
    pdf_files = sorted(p for p in pdf_dir.glob("*.pdf") if p.is_file())

    all_chunks: list[ChunkData] = []

    for pdf_path in pdf_files:
        chunks = load_pdf(pdf_path)
        all_chunks.extend(chunks)

    return all_chunks