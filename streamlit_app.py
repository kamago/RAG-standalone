# app.py
# Streamlit RAG demo using tabular (CSV) input as the knowledge base, with chunking.
#
# Requirements (requirements.txt):
# streamlit
# pandas
# numpy
# faiss-cpu
# sentence-transformers
# openai
#
# Secrets (.streamlit/secrets.toml):
# OPENAI_API_KEY="..."

import os
import numpy as np
import pandas as pd
import streamlit as st

try:
    import faiss  # faiss-cpu
except Exception:
    faiss = None

from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -----------------------------
# Page config + UI chrome
# -----------------------------
st.set_page_config(page_title="Tabular RAG (CSV → Chunking → Retrieval → Answer)", layout="wide")
st.title("📊 Tabular RAG (with chunking)")
st.caption("Upload a CSV, chunk + index selected columns, retrieve relevant chunks, then answer with an LLM.")

with st.sidebar:
    st.header("1) Upload data")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    st.header("2) RAG settings")
    embed_model_name = st.selectbox(
        "Embedding model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L12-v2",
        ],
        index=0,
    )
    top_k = st.slider("Retrieved chunks (top-k)", 1, 20, 5)

    st.subheader("Chunking")
    chunk_size = st.number_input("Chunk size (characters)", min_value=200, max_value=3000, value=800, step=100)
    chunk_overlap = st.number_input("Chunk overlap (characters)", min_value=0, max_value=1000, value=120, step=20)

    st.header("3) LLM settings")
    model_name = st.selectbox("LLM model", ["gpt-4o-mini", "gpt-4.1-mini"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

# -----------------------------
# Helpers
# -----------------------------
def safe_get_openai_key() -> str | None:
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return os.getenv("OPENAI_API_KEY")

@st.cache_resource
def load_embedder(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)

def make_row_text(df: pd.DataFrame, cols: list[str], row: pd.Series) -> str:
    """Convert a single row into a clean 'document' string."""
    parts = []
    for c in cols:
        val = row.get(c)
        if pd.isna(val):
            continue
        parts.append(f"{c}: {str(val)}")
    return " | ".join(parts)

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """
    Simple character-based chunker with overlap.
    - Good enough for a beginner RAG and works for any kind of text.
    """
    text = (text or "").strip()
    if not text:
        return []

    if size <= 0:
        return [text]

    if overlap >= size:
        overlap = max(0, size // 4)  # avoid infinite loop

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap  # overlap
        if start < 0:
            start = 0

    return chunks

def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS index for cosine similarity via inner product on normalized vectors."""
    if faiss is None:
        raise RuntimeError("faiss is not installed. Add 'faiss-cpu' to requirements.txt")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine sim
    index.add(embeddings.astype(np.float32))
    return index

def retrieve(query: str, embedder: SentenceTransformer, index, chunk_records: list[dict], k: int):
    q_emb = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q_emb, k)

    hits = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx == -1:
            continue
        rec = chunk_records[int(idx)]
        hits.append(
            {
                "score": float(score),
                "chunk_text": rec["chunk_text"],
                "row_id": rec["row_id"],
                "chunk_id": rec["chunk_id"],
            }
        )
    return hits

def generate_answer(client: OpenAI, model: str, question: str, contexts: list[dict], temperature: float):
    # Build a compact context block
    context_text = "\n".join(
        [
            f"- (row {c['row_id']}, chunk {c['chunk_id']}) {c['chunk_text']}"
            for c in contexts
        ]
    )

    system = (
        "You are a helpful assistant answering questions using ONLY the provided context. "
        "If the context doesn't contain the answer, say you don't know and suggest what data would be needed."
    )
    user = f"""Question:
{question}

Context (retrieved chunks):
{context_text}

Instructions:
- Answer concisely.
- If you cite facts, mention which row/chunk supports it.
"""

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return resp.output_text

# -----------------------------
# Main app flow
# -----------------------------
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Preview")
st.dataframe(df.head(25), use_container_width=True)

if df.empty:
    st.warning("Your CSV is empty.")
    st.stop()

# Select columns to index
st.subheader("Select columns to use as knowledge")
all_cols = list(df.columns)
default_cols = all_cols[: min(4, len(all_cols))]
cols_to_index = st.multiselect("Columns to index", all_cols, default=default_cols)

# Choose an ID column (optional but helpful for grounding)
id_candidates = ["ORDER_ID", "order_id", "ID", "id"]
default_id = next((c for c in id_candidates if c in df.columns), None)
row_id_col = st.selectbox(
    "Row identifier column (optional, helps trace sources)",
    options=["(use row number)"] + all_cols,
    index=(1 + all_cols.index(default_id)) if default_id in all_cols else 0,
)

if not cols_to_index:
    st.warning("Select at least one column to index.")
    st.stop()

# Build chunked docs + index (cached)
@st.cache_resource(show_spinner=False)
def build_rag_assets(df_in: pd.DataFrame, cols: tuple[str, ...], embed_model: str, chunk_size: int, chunk_overlap: int, row_id_col: str):
    embedder = load_embedder(embed_model)

    chunk_records: list[dict] = []
    for i in range(len(df_in)):
        row = df_in.iloc[i]
        row_text = make_row_text(df_in, list(cols), row)
        chunks = chunk_text(row_text, size=chunk_size, overlap=chunk_overlap)

        # Determine row id
        if row_id_col != "(use row number)":
            rid = row.get(row_id_col)
            rid = str(rid) if not pd.isna(rid) else str(i)
        else:
            rid = str(i)

        for j, ch in enumerate(chunks):
            chunk_records.append(
                {
                    "row_id": rid,
                    "chunk_id": j,
                    "chunk_text": ch,
                }
            )

    # Embedding all chunk texts
    texts = [r["chunk_text"] for r in chunk_records]
    if not texts:
        raise RuntimeError("No text found to index after chunking. Check your selected columns.")

    embeddings = embedder.encode(texts, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    index = build_faiss_index(embeddings)
    return embedder, chunk_records, index

with st.spinner("Building chunked vector index (embeddings + FAISS)…"):
    try:
        embedder, chunk_records, index = build_rag_assets(
            df,
            tuple(cols_to_index),
            embed_model_name,
            int(chunk_size),
            int(chunk_overlap),
            row_id_col,
        )
    except Exception as e:
        st.error("Failed to build the RAG index.")
        st.exception(e)
        st.stop()

st.success(f"Indexed {len(chunk_records):,} chunks from {len(df):,} rows.")

# Query UI
st.subheader("Ask a question")
question = st.text_input("Question", placeholder="e.g., What issues mention late delivery or damaged packaging?")

ask = st.button("Retrieve + Answer", type="primary")

if not ask:
    st.stop()

if not question.strip():
    st.warning("Please enter a question.")
    st.stop()

# Retrieve
with st.spinner("Retrieving relevant chunks…"):
    hits = retrieve(question, embedder, index, chunk_records, int(top_k))

st.markdown("### Retrieved chunks")
if not hits:
    st.warning("No relevant chunks found.")
    st.stop()

hits_df = pd.DataFrame(
    [
        {
            "rank": i + 1,
            "score": h["score"],
            "row_id": h["row_id"],
            "chunk_id": h["chunk_id"],
            "chunk_text": h["chunk_text"],
        }
        for i, h in enumerate(hits)
    ]
)
st.dataframe(hits_df, use_container_width=True)

# Generate answer with OpenAI (optional)
api_key = safe_get_openai_key()
if not api_key:
    st.warning(
        "No OPENAI_API_KEY found. Retrieval works, but I can’t generate an LLM answer.\n\n"
        "Add OPENAI_API_KEY to `.streamlit/secrets.toml` or set it as an environment variable."
    )
    st.stop()

client = OpenAI(api_key=api_key)

with st.spinner("Generating answer with the LLM…"):
    try:
        answer = generate_answer(
            client=client,
            model=model_name,
            question=question,
            contexts=hits,
            temperature=temperature,
        )
    except Exception as e:
        st.error("LLM generation failed.")
        st.exception(e)
        st.stop()

st.markdown("### Answer")
st.write(answer)

with st.expander("What chunking is doing here?"):
    st.write(
        "- Each row is turned into one text string from your selected columns.\n"
        f"- That string is split into chunks of ~{int(chunk_size)} characters with {int(chunk_overlap)} characters overlap.\n"
        "- We embed and index chunks, not whole rows, so retrieval is more precise and avoids super-long embeddings."
    )
