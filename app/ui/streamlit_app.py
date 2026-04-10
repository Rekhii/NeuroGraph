"""
Streamlit UI — app/ui/streamlit_app.py

Professional research interface for NeuroGraph.
Monochrome scientific aesthetic with handwritten title.
Animated neural particle network background.

Usage:
    streamlit run app/ui/streamlit_app.py
"""

import streamlit as st
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Caveat:wght@400;500;600;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-void: #0a0a0a;
    --bg-surface: #111111;
    --bg-card: #161616;
    --bg-elevated: #1a1a1a;
    --border: rgba(255, 255, 255, 0.06);
    --border-hover: rgba(255, 255, 255, 0.12);
    --border-accent: rgba(255, 255, 255, 0.25);
    --white: #f0f0f0;
    --white-mid: #a0a0a0;
    --white-dim: #666666;
    --white-ghost: #3a3a3a;
    --font-title: 'Caveat', cursive;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

.stApp { background: var(--bg-void) !important; color: var(--white) !important; }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: var(--bg-surface) !important; border-right: 1px solid var(--border); }

.ng-header {
    position: relative; margin: -1rem -1rem 2.5rem -1rem;
    height: 200px; overflow: hidden;
    border-bottom: 1px solid var(--border); background: var(--bg-void);
}
.ng-header canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
.ng-header-content {
    position: relative; z-index: 2; display: flex; flex-direction: column;
    justify-content: center; align-items: center; height: 100%; text-align: center;
}
.ng-title {
    font-family: var(--font-title); font-weight: 700; font-size: 4.2rem;
    color: var(--white); margin: 0; letter-spacing: 0.02em; line-height: 1;
    text-shadow: 0 0 60px rgba(255,255,255,0.08);
}
.ng-tagline {
    font-family: var(--font-body); font-weight: 300; font-size: 0.78rem;
    color: var(--white-dim); letter-spacing: 0.25em; text-transform: uppercase; margin-top: 0.8rem;
}
.ng-tagline span { color: var(--white-mid); font-weight: 500; }

.sec {
    font-family: var(--font-body); font-weight: 500; font-size: 0.68rem;
    letter-spacing: 0.2em; text-transform: uppercase; color: var(--white-dim);
    padding-bottom: 0.6rem; border-bottom: 1px solid var(--border); margin: 2rem 0 1rem 0;
}

/* Answer container — styles st.container(border=True) used by render_answer */
[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-left: 2px solid var(--white-dim) !important;
    border-radius: 4px !important;
    padding: 1.2rem 1.5rem !important;
}

.src-item {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 3px;
    padding: 0.5rem 0.9rem; margin: 0.25rem 0.3rem 0.25rem 0;
    font-family: var(--font-mono); font-size: 0.75rem; color: var(--white-mid);
    transition: border-color 0.2s;
}
.src-item:hover { border-color: var(--border-hover); }
.src-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--white-dim); }

.met-row {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
    background: var(--border); border: 1px solid var(--border);
    border-radius: 4px; overflow: hidden; margin: 1.2rem 0 2rem 0;
}
.met { background: var(--bg-card); padding: 1.2rem 1rem; text-align: center; }
.met-val { font-family: var(--font-mono); font-weight: 500; font-size: 1.4rem; color: var(--white); line-height: 1; }
.met-lab { font-family: var(--font-body); font-size: 0.62rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--white-ghost); margin-top: 0.5rem; }

.chunk { background: var(--bg-surface); border: 1px solid var(--border); border-radius: 3px; padding: 1rem 1.2rem; margin: 0.4rem 0; }
.chunk-head { font-family: var(--font-mono); font-size: 0.72rem; color: var(--white-dim); margin-bottom: 0.5rem; display: flex; gap: 0.8rem; align-items: center; }
.chunk-score { background: rgba(255,255,255,0.05); padding: 0.1rem 0.45rem; border-radius: 2px; color: var(--white-mid); font-weight: 500; }
.chunk-body { font-family: var(--font-body); font-size: 0.84rem; color: var(--white-dim); line-height: 1.7; }

.f-item { background: var(--bg-surface); border: 1px solid var(--border); border-radius: 3px; padding: 0.55rem 0.9rem; margin: 0.25rem 0; font-family: var(--font-mono); font-size: 0.78rem; color: var(--white-mid); display: flex; justify-content: space-between; }
.f-item .f-sz { color: var(--white-ghost); }

.st-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; margin-right: 0.4rem; }
.st-on { background: #4ade80; box-shadow: 0 0 6px rgba(74,222,128,0.4); }
.st-off { background: #666; }

.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { font-family: var(--font-body); font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; font-size: 0.72rem; color: var(--white-ghost); padding: 0.8rem 1.8rem; }
.stTabs [aria-selected="true"] { color: var(--white) !important; border-bottom-color: var(--white) !important; }

.stButton > button[kind="primary"] { background: var(--white) !important; color: var(--bg-void) !important; border: none !important; font-family: var(--font-body); font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; font-size: 0.72rem; padding: 0.65rem 1.8rem; border-radius: 3px; transition: opacity 0.2s; }
.stButton > button[kind="primary"]:hover { opacity: 0.85; }
.stButton > button:not([kind="primary"]) { background: transparent !important; border: 1px solid var(--border-hover) !important; color: var(--white-mid) !important; font-family: var(--font-body); font-size: 0.72rem; letter-spacing: 0.04em; border-radius: 3px; }

.stTextArea textarea { background: var(--bg-card) !important; border: 1px solid var(--border) !important; color: var(--white) !important; font-family: var(--font-body) !important; font-size: 0.9rem !important; border-radius: 4px !important; }
.stTextArea textarea:focus { border-color: var(--border-accent) !important; box-shadow: none !important; }
.stTextArea textarea::placeholder { color: var(--white-ghost) !important; font-style: italic; }

.streamlit-expanderHeader { font-family: var(--font-body); font-weight: 500; font-size: 0.78rem; letter-spacing: 0.06em; color: var(--white-dim); }
[data-testid="stMetric"] { background: transparent; }
[data-testid="stMetricLabel"] { font-family: var(--font-body) !important; font-size: 0.68rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; color: var(--white-ghost) !important; }
[data-testid="stMetricValue"] { font-family: var(--font-mono) !important; color: var(--white-mid) !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--white-ghost); border-radius: 2px; }
</style>
"""

PARTICLE_JS = """
<script>
(function() {
    const c = document.getElementById('ng-cv');
    if (!c) return;
    const ctx = c.getContext('2d');
    let w, h, pts = [];
    const N = 60, D = 130, S = 0.25;
    function resize() { const r = c.parentElement.getBoundingClientRect(); w = c.width = r.width; h = c.height = r.height; }
    function init() { resize(); pts = []; for (let i = 0; i < N; i++) pts.push({ x: Math.random()*w, y: Math.random()*h, vx: (Math.random()-0.5)*S, vy: (Math.random()-0.5)*S, r: Math.random()*1.5+0.3, p: Math.random()*Math.PI*2 }); }
    function draw() {
        ctx.clearRect(0, 0, w, h);
        for (let i = 0; i < pts.length; i++) for (let j = i+1; j < pts.length; j++) {
            const dx = pts[i].x-pts[j].x, dy = pts[i].y-pts[j].y, d = Math.sqrt(dx*dx+dy*dy);
            if (d < D) { ctx.strokeStyle = `rgba(255,255,255,${(1-d/D)*0.12})`; ctx.lineWidth = 0.5; ctx.beginPath(); ctx.moveTo(pts[i].x, pts[i].y); ctx.lineTo(pts[j].x, pts[j].y); ctx.stroke(); }
        }
        for (const p of pts) { p.p += 0.015; ctx.fillStyle = `rgba(255,255,255,${0.25+Math.sin(p.p)*0.15})`; ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI*2); ctx.fill(); p.x += p.vx; p.y += p.vy; if (p.x<0||p.x>w) p.vx*=-1; if (p.y<0||p.y>h) p.vy*=-1; }
        requestAnimationFrame(draw);
    }
    init(); draw(); window.addEventListener('resize', resize);
})();
</script>
"""


def check_ollama(base_url):
    try:
        import httpx
        resp = httpx.get(f"{base_url}/api/tags", timeout=3.0)
        if resp.status_code == 200:
            return True, [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass
    return False, []


def save_uploaded_pdfs(uploaded_files, raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for f in uploaded_files:
        if f.name.lower().endswith(".pdf"):
            (raw_dir / f.name).write_bytes(f.getbuffer())
            saved += 1
    return saved


def index_pdfs(incremental=True):
    from app.rag.loader import load_all_pdfs
    from app.rag.embedder import VectorStore

    chunks = load_all_pdfs()
    if not chunks:
        return 0, 0

    store = VectorStore()

    if incremental:
        existing_ids = store.get_indexed_ids()
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        if new_chunks:
            store.add_chunks(new_chunks)
    else:
        store.add_chunks(chunks)

    num_sources = len(set(c["source"] for c in chunks))
    return store.count, num_sources


def get_chunk_count():
    try:
        from app.rag.embedder import get_store
        return get_store().count
    except Exception:
        return 0


def list_indexed_pdfs():
    try:
        from app.core.config import settings
        settings.data_raw.mkdir(parents=True, exist_ok=True)
        return sorted(p.name for p in settings.data_raw.glob("*.pdf"))
    except Exception:
        return []


def render_header():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="ng-header">
        <canvas id="ng-cv"></canvas>
        <div class="ng-header-content">
            <h1 class="ng-title">NeuroGraph</h1>
            <div class="ng-tagline">the <span>researcher</span> &mdash; multi agent retrieval augmented generation</div>
        </div>
    </div>
    {PARTICLE_JS}
    """, unsafe_allow_html=True)


def render_metrics(confidence, num_sources, time_ms, num_errors):
    st.markdown(f"""
    <div class="met-row">
        <div class="met"><div class="met-val">{confidence:.0%}</div><div class="met-lab">Confidence</div></div>
        <div class="met"><div class="met-val">{num_sources}</div><div class="met-lab">Sources</div></div>
        <div class="met"><div class="met-val">{time_ms/1000:.1f}s</div><div class="met-lab">Latency</div></div>
        <div class="met"><div class="met-val">{num_errors}</div><div class="met-lab">Errors</div></div>
    </div>
    """, unsafe_allow_html=True)


def render_answer(text):
    with st.container(border=True):
        st.markdown(text)


def render_sources(sources):
    html_parts = ""
    for s in sources:
        if isinstance(s, dict):
            pages = ", ".join(str(p) for p in s.get("pages", []))
            html_parts += f'<span class="src-item"><span class="src-dot"></span>{s.get("source", "?")} pp. {pages}</span>'
        else:
            html_parts += f'<span class="src-item"><span class="src-dot"></span>{s}</span>'
    st.markdown(html_parts, unsafe_allow_html=True)


def render_chunk(idx, r):
    st.markdown(f"""
    <div class="chunk">
        <div class="chunk-head">
            <span>[{idx}] {r.source} p.{r.page}</span>
            <span class="chunk-score">{r.effective_score:.3f}</span>
            <span>{r.strategy}</span>
        </div>
        <div class="chunk-body">{r.text[:400]}{"..." if len(r.text) > 400 else ""}</div>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="NeuroGraph", page_icon="N", layout="wide", initial_sidebar_state="expanded")
    render_header()

    from app.core.config import settings
    try:
        settings.data_raw.mkdir(parents=True, exist_ok=True)
        settings.data_processed.mkdir(parents=True, exist_ok=True)
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    with st.sidebar:
        ollama_ok, _ = check_ollama(settings.ollama_base_url)
        dot = "st-on" if ollama_ok else "st-off"
        label = "Ollama" if ollama_ok else "Ollama offline"
        st.markdown(f'<p style="font-size:0.78rem;color:var(--white-dim)"><span class="st-dot {dot}"></span>{label}</p>', unsafe_allow_html=True)

        st.metric("Knowledge base", f"{get_chunk_count()} chunks")
        pdfs = list_indexed_pdfs()
        if pdfs:
            with st.expander(f"{len(pdfs)} papers"):
                for name in pdfs:
                    st.caption(name)

        st.markdown("---")
        strategy = st.selectbox("Strategy", ["auto", "hybrid", "vector", "keyword"], index=0)
        top_k = st.slider("Depth", 1, 30, 15)
        enable_rerank = st.checkbox("Reranking", value=True)
        enable_ctx = st.checkbox("Context optimization", value=True)

    tab_q, tab_d = st.tabs(["QUERY", "DOCUMENTS"])

    with tab_d:
        st.markdown('<div class="sec">Upload research papers</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

        if uploaded:
            for f in uploaded:
                sz = len(f.getbuffer()) / 1024
                u, v = ("KB", sz) if sz < 1024 else ("MB", sz / 1024)
                st.markdown(f'<div class="f-item"><span>{f.name}</span><span class="f-sz">{v:.1f} {u}</span></div>', unsafe_allow_html=True)

            if st.button("Index documents", type="primary"):
                with st.spinner("Saving..."):
                    saved = save_uploaded_pdfs(uploaded, settings.data_raw)
                if saved:
                    with st.spinner("Indexing..."):
                        try:
                            count, ns = index_pdfs()
                            st.success(f"{count} chunks from {ns} documents")
                        except Exception as e:
                            st.error(str(e))

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Rebuild index"):
                with st.spinner("Rebuilding..."):
                    try:
                        from app.rag.embedder import get_store
                        get_store().reset()
                        count, ns = index_pdfs(incremental=False)
                        st.success(f"{count} chunks / {ns} docs")
                    except Exception as e:
                        st.error(str(e))
        with c2:
            if st.button("Clear all"):
                try:
                    from app.rag.embedder import get_store
                    get_store().reset()
                    for p in settings.data_raw.glob("*.pdf"): p.unlink()
                    st.success("Cleared")
                except Exception as e:
                    st.error(str(e))

    with tab_q:
        cc = get_chunk_count()
        if cc == 0:
            st.info("Knowledge base empty. Upload papers in DOCUMENTS tab.")

        query = st.text_area("q", placeholder="How does active inference differ from Global Workspace Theory?", height=80, label_visibility="collapsed")

        if st.button("Execute", type="primary") and query.strip():
            if cc == 0:
                st.error("No documents indexed.")
                return
            with st.spinner("Retrieving evidence..."):
                try:
                    from app.pipeline import Pipeline
                    pipe = Pipeline(enable_rerank=enable_rerank, enable_context_optimization=enable_ctx)
                    t0 = time.monotonic()
                    result = pipe.run(query=query, strategy=strategy if strategy != "auto" else None, top_k=top_k)
                    tt = (time.monotonic() - t0) * 1000
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    with st.expander("Trace"): st.code(traceback.format_exc())
                    return

            render_metrics(result.confidence, len(result.sources), tt, len(result.errors))

            st.markdown('<div class="sec">Response</div>', unsafe_allow_html=True)
            render_answer(result.answer)

            if result.sources:
                st.markdown('<div class="sec">References</div>', unsafe_allow_html=True)
                render_sources(result.sources)

            with st.expander("Planning trace"):
                if result.plan: st.json(result.plan.to_dict())

            with st.expander("Retrieval trace"):
                if result.retrieval:
                    st.markdown(f"`{result.retrieval.strategy}` · `{len(result.retrieval.results)} results` · `top {result.retrieval.top_score:.4f}`")
                    if result.retrieval.sub_queries: st.markdown(f"Sub-queries: `{result.retrieval.sub_queries}`")
                    for i, r in enumerate(result.retrieval.results, 1): render_chunk(i, r)

            with st.expander("Critic verdict"):
                if result.verdict: st.json(result.verdict.to_dict())

            with st.expander("Performance"):
                if result.timing:
                    for s, ms in result.timing.items(): st.markdown(f"`{s}` — {ms:.0f}ms")
                    st.markdown(f"**Total** — {tt:.0f}ms")

            if result.errors:
                with st.expander("Errors"):
                    for e in result.errors: st.warning(e)


if __name__ == "__main__":
    main()