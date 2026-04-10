# NeuroGraph

**Multi-Agent RAG system for Deep Researcher.**

Upload complex research papers. Ask questions. Get detailed, cited answers with proper LaTeX mathematics, term-by-term equation breakdowns, and full derivations — grounded entirely in your uploaded documents.

![System Architecture](docs/architecture.png)

---

## What it does

NeuroGraph is not a summarizer. It is a research tutor that reads your papers and teaches you what's inside them.

Drop a 400-page textbook on theoretical neuroscience, a dense machine learning paper, or a graduate-level math reference. Ask it to derive an equation, compare two frameworks, or explain a mechanism. It retrieves the relevant passages, checks whether the evidence is sufficient, and writes a detailed pedagogical answer with inline citations and rendered LaTeX.

**Example query:**
> "Derive the scaled dot-product attention formula step by step. Start from the definition of queries, keys, and values as linear projections, show why we scale by the square root of d_k using the variance argument, and explain how multi-head attention extends single-head attention."

**What you get back:**
- Full derivation with every equation in rendered LaTeX
- Term-by-term breakdown of each variable
- Matrix dimensions at every step
- Inline citations `[1]`, `[3]`, `[7]` linking to specific pages in your uploaded papers
- Limitations section noting what the evidence did not cover

---

## Demo

> [Watch the demo video](https://youtube.com/your-link-here)

---

## Architecture

The system runs a 5-stage pipeline with a critic feedback loop:

```
User query
  → Planner agent (classify, decompose, set retrieval strategy)
  → Retriever (hybrid vector + BM25, cross-encoder reranking)
  → Critic agent (evaluate evidence, re-retrieve if insufficient)
  → Context optimizer (filter, compress, enforce token budget)
  → Writer agent (generate cited answer with LaTeX)
  → Cited answer + sources
```

**The critic feedback loop** is the architectural differentiator. If the critic determines that retrieved evidence is insufficient — missing one side of a comparison, lacking the specific equation requested, or scoring below the confidence threshold — it generates a refined query and triggers re-retrieval with previously seen chunks excluded. This loop runs up to 3 iterations before accepting the best available evidence with caveats.

### Pipeline components

| Component | File | What it does |
|---|---|---|
| Config | `app/core/config.py` | Centralized settings, chunk size, model selection |
| LLM Client | `app/core/llm.py` | Unified Ollama/Groq client with retry, token tracking |
| Loader | `app/rag/loader.py` | Math-aware PDF extraction, preserves equation structure |
| Embedder | `app/rag/embedder.py` | Batched embedding with validation, ChromaDB storage |
| Retriever | `app/rag/retriever.py` | Hybrid search, BM25, cross-encoder reranking, feedback loop |
| Context | `app/rag/context.py` | Relevance filtering, redundancy removal, sentence extraction, token budget |
| Planner | `app/agents/planner.py` | Query classification, decomposition, strategy routing |
| Critic | `app/agents/critic.py` | Evidence evaluation, gap detection, re-retrieval triggering |
| Writer | `app/agents/writer.py` | LaTeX-formatted cited answers with term-by-term breakdowns |
| Pipeline | `app/pipeline.py` | Orchestrator wiring all stages with timing and error handling |
| API | `app/api/main.py` | FastAPI REST endpoints |
| UI | `app/ui/streamlit_app.py` | Streamlit interface with particle animation header |

---

## Key features

### Multi-agent pipeline
Four specialized agents (planner, retriever, critic, writer) with distinct roles. The planner classifies queries and sets retrieval strategy. The critic evaluates evidence quality and triggers re-retrieval when insufficient. The writer produces pedagogical answers grounded strictly in retrieved evidence.

### Hybrid retrieval
Combines vector similarity search (ChromaDB) with BM25 keyword scoring. Vector search captures semantic meaning ("consciousness" matches "awareness"). BM25 catches exact terms, acronyms, and equation references that embedding models miss. Results are merged, deduplicated, and reranked using a cross-encoder model.

### Cross-encoder reranking
After initial retrieval, a cross-encoder (`ms-marco-MiniLM-L-6-v2`, 22M params, runs on CPU) rescores each query-document pair jointly. This catches relevance nuances that independent embedding comparison misses. Runs locally, no API key needed.

### Math-aware extraction
PDF text extraction preserves equation structure instead of collapsing whitespace. Lines containing mathematical symbols (`=`, `∑`, `∫`, Greek letters) are kept intact. Chunks are tagged with `has_math` metadata so retrieval can prioritize mathematical content for theory-heavy queries.

### LaTeX rendering
The writer agent is instructed to reproduce all mathematics in proper LaTeX format (`$$...$$` for block equations, `$...$` for inline). Every equation is followed by a term-by-term breakdown explaining what each variable represents, what the equation computes, and why it matters.

### Context optimization
A 4-stage pipeline between retriever and writer: relevance gate (drop low-score chunks), redundancy filter (Jaccard similarity dedup), sentence extractor (keep only query-relevant sentences), and token budget enforcer. Reduces context size by 30-50% while preserving signal.

### Critic feedback loop
The critic evaluates evidence across four dimensions: score quality, keyword coverage, volume sufficiency, and comparison balance. If evidence is insufficient, it generates a refined query and triggers re-retrieval with previously seen chunks excluded. On the final iteration, it accepts best-available evidence and adds writer caveats.

### Adaptive routing
Queries are classified into four types (factual, conceptual, comparison, multi-step) and routed to the optimal retrieval strategy. Math-heavy queries are automatically boosted to hybrid search. The planner provides LLM-powered routing when available, with a rule-based fallback.

### Incremental indexing
New PDFs are indexed without re-embedding the entire corpus. The system checks which chunk IDs already exist in ChromaDB and only embeds new chunks. Full reindexing is available via the "Rebuild index" button.

### Dual LLM providers
Supports Ollama (local, free, offline) and Groq (cloud, free tier, 14,400 requests/day). Switch between them with one environment variable. Both use the same OpenAI-compatible API, so no code changes needed.

### Graceful degradation
Every agent has a rule-based fallback. If the LLM is unavailable, the planner uses regex classification, the critic uses heuristic scoring, and the writer assembles answers from raw chunks. The system never crashes — it degrades to lower quality but still functions.

---

## Tech stack

All free. No API costs.

| Layer | Technology | Purpose |
|---|---|---|
| LLM | Ollama (qwen3:8b) or Groq | Reasoning, planning, writing |
| Embeddings | nomic-embed-text via Ollama | Text to vector conversion |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Passage reranking (CPU) |
| Vector DB | ChromaDB | Persistent vector storage |
| Keyword search | rank-bm25 | BM25 scoring |
| HTTP client | httpx | LLM API communication |
| API | FastAPI + uvicorn | REST endpoints |
| UI | Streamlit | Demo interface |
| PDF extraction | pypdf | Text extraction from PDFs |
| Text splitting | LangChain text splitters | Recursive chunking |
| Containerization | Docker | Deployment |

---

## Setup

### Prerequisites
- Python 3.11+
- 8GB+ RAM (for Ollama models)

### Install Ollama and models

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

### Install dependencies

```bash
git clone https://github.com/yourusername/neurograph.git
cd neurograph
pip install -r requirements.txt
```

### Configure environment

```bash
cp .env.example .env
# Edit .env if using Groq instead of Ollama
```

### Run

```bash
streamlit run app/ui/streamlit_app.py
```

Or run the API:

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

---

## Usage

### 1. Upload papers
Switch to the **DOCUMENTS** tab. Upload one or more PDFs. Click **Index documents**.

### 2. Ask questions
Switch to the **QUERY** tab. Type your question. Click **Execute**.

### 3. Explore traces
Expand the debug panels below the answer to see the planner's classification, retrieval scores, critic verdict, and per-stage timing.

### Example queries

**Factual:**
> "What is the dimension of the key vectors in the Transformer model?"

**Conceptual:**
> "Explain how the NEF represents a continuous variable using a population of spiking neurons. Show the encoding equation and explain each term."

**Comparison:**
> "Compare scaled dot-product attention with additive attention. Show the mathematical formulation of each and explain the computational trade-offs."

**Multi-step derivation:**
> "Derive the optimal linear decoding weights in the Neural Engineering Framework by minimizing mean squared error. Show the full matrix formulation and connect it to the least-squares solution."

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/query` | Run the full pipeline on a question |
| POST | `/index` | Index PDFs from data/raw/ |
| GET | `/health` | Health check (LLM + vector store status) |
| GET | `/stats` | Chunk count, token usage, config |

### Example API call

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does multi-head attention work?", "top_k": 15}'
```

---

## Project structure

```
neurograph/
├── app/
│   ├── core/
│   │   ├── config.py          # Centralized settings
│   │   └── llm.py             # Unified LLM client
│   ├── rag/
│   │   ├── loader.py          # Math-aware PDF extraction
│   │   ├── embedder.py        # Embedding + ChromaDB storage
│   │   ├── retriever.py       # Hybrid search + reranking
│   │   └── context.py         # Context optimization
│   ├── agents/
│   │   ├── planner.py         # Query classification + routing
│   │   ├── critic.py          # Evidence evaluation + feedback
│   │   └── writer.py          # LaTeX answer generation
│   ├── api/
│   │   └── main.py            # FastAPI endpoints
│   ├── ui/
│   │   └── streamlit_app.py   # Streamlit interface
│   └── pipeline.py            # Pipeline orchestrator
├── data/
│   ├── raw/                   # Drop PDFs here
│   ├── processed/             # Chunked documents
│   └── index/chroma/          # Vector database
├── docs/
│   └── architecture.png       # System architecture diagram
├── requirements.txt
├── Dockerfile
├── .env.example
├── .gitignore
└── README.md
```

---

## Design decisions

**Why hybrid retrieval over pure vector search?**
Vector search misses exact terms. A query about "Eq. 3.46" or "COGITATE" needs keyword matching. BM25 catches these while vector search captures semantic meaning. Combining both covers more ground.

**Why a critic agent instead of just retrieving and writing?**
Without the critic, the writer hallucinates when evidence is weak. The critic checks coverage before the writer sees anything. If evidence is insufficient, re-retrieval finds better chunks. This is the anti-hallucination architecture.

**Why rule-based fallbacks for every agent?**
The system must work without an LLM running. During demos, if Ollama crashes or Groq rate-limits, the planner still classifies queries, the critic still evaluates evidence, and the writer still produces cited answers. Degraded quality, but never a crash.

**Why larger chunk sizes (1000 chars) for math papers?**
Standard 500-char chunks split equations across boundaries. A 1000-char chunk with 200-char overlap keeps equations intact with their surrounding explanatory text. This is critical for retrieval quality on theory-heavy papers.

**Why cross-encoder reranking on CPU?**
The ms-marco-MiniLM model is 22M parameters and scores 20 candidates in under 1 second on CPU. No GPU needed. The quality improvement over raw embedding similarity is significant — it catches cases where a chunk discusses the right topic but doesn't answer the specific question.

---


## License

MIT

---
