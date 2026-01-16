# app.py
# Streamlit RAG demo using *tabular input data* (CSV) as the knowledge base.
# - Upload a CSV
# - Choose which columns to index as text
# - Build an in-memory vector index (FAISS) using Sentence-Transformers embeddings
# - Ask questions; the app retrieves the most relevant rows and uses an LLM to answer
#
# Works best with Streamlit Community Cloud or local. For the LLM, this example uses OpenAI.
# Put your key in .streamlit/secrets.toml:
#   OPENAI_API_KEY="..."
#
# Requirements (requirements.txt):
# streamlit
# pandas
# numpy
# faiss-cpu
# sentence-transformers
# openai

import os
import numpy as np
import pandas as pd
import streamlit as st

try:
    import faiss  # faiss-cpu
except Exception as e:
    faiss = None

from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -----------------------------
# Page config + UI chrome
# -----------------------------
st.set_page_config(page_title="Tabular RAG (CSV → Retrieval → Answer)", layout="wide")
st.title("📊 Tabular RAG: Ask questions over your CSV")
st.caption("Upload a CSV, index selected columns, retrieve the most relevant rows, then generate an answer with an LLM.")

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
    top_k = st.slider("Retrieved rows (top-k)", 1, 20, 5)

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

def make_row_text(df: pd.DataFrame, cols: list[str], row: pd.Series) -> str:
    """Convert a single row into a clean 'document' string."""
    parts = []
    for c in cols:
        val = row.get(c)
        if pd.isna(val):
            continue
        parts.append(f"{c}: {str(val)}")
    return " | ".join(parts)

@st.cache_resource
def load_embedder(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)

def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS index for cosine similarity via inner product on normalized vectors."""
    if faiss is None:
        raise RuntimeError("faiss is not installed. Add 'faiss-cpu' to requirements.txt")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product
    index.add(embeddings.astype(np.float32))
    return index

def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

def retrieve(query: str, embedder: SentenceTransformer, index, docs: list[str], k: int):
    q_emb = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q_emb, k)
    hits = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx == -1:
            continue
        hits.append({"score": float(score), "doc": docs[idx], "doc_id": int(idx)})
    return hits

def generate_answer(client: OpenAI, model: str, question: str, contexts: list[dict], temperature: float):
    context_text = "\n".join([f"- {c['doc']}" for c in contexts])

    system = (
        "You are a helpful assistant answering questions using ONLY the provided context. "
        "If the context doesn't contain the answer, say you don't know and suggest what data would be needed."
    )
    user = f"""Question:
{question}

Context (retrieved rows):
{context_text}

Instructions:
- Answer concisely.
- If you cite facts, reference the relevant row content.
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

# Load data
df = pd.read_csv(uploaded)
st.subheader("Preview")
st.dataframe(df.head(25), use_container_width=True)

if df.empty:
    st.warning("Your CSV is empty.")
    st.stop()

# Select columns to index as text
st.subheader("Select columns to use as knowledge")
all_cols = list(df.columns)
default_cols = all_cols[: min(4, len(all_cols))]
cols_to_index = st.multiselect("Columns to index", all_cols, default=default_cols)

if not cols_to_index:
    st.warning("Select at least one column to index.")
    st.stop()

# Build docs
with st.expander("How rows are converted to text", expanded=False):
    st.write("Each row becomes a single text chunk like:")
    example = make_row_text(df, cols_to_index, df.iloc[0])
    st.code(example)

# Build embeddings + FAISS index (cached per file content + settings)
@st.cache_resource(show_spinner=False)
def build_rag_assets(df_in: pd.DataFrame, cols: tuple[str, ...], embed_model: str):
    embedder = load_embedder(embed_model)

    docs = [make_row_text(df_in, list(cols), df_in.iloc[i]) for i in range(len(df_in))]
    embeddings = embedder.encode(docs, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    index = build_faiss_index(embeddings)
    return embedder, docs, index

with st.spinner("Building vector index (embeddings + FAISS)…"):
    try:
        embedder, docs, index = build_rag_assets(df, tuple(cols_to_index), embed_model_name)
    except Exception as e:
        st.error("Failed to build the RAG index.")
        st.exception(e)
        st.stop()

# Query UI
st.subheader("Ask a question")
question = st.text_input("Question", placeholder="e.g., What are the most common complaints?")

colA, colB = st.columns([1, 2])
with colA:
    ask = st.button("Retrieve + Answer", type="primary")
with colB:
    st.caption("Tip: ask specific questions that match your table columns for best retrieval.")

if not ask:
    st.stop()

if not question.strip():
    st.warning("Please enter a question.")
    st.stop()

# Retrieve
with st.spinner("Retrieving relevant rows…"):
    hits = retrieve(question, embedder, index, docs, top_k)

st.markdown("### Retrieved rows")
if not hits:
    st.warning("No relevant rows found.")
    st.stop()

hits_df = pd.DataFrame(
    [{"rank": i + 1, "score": h["score"], "row_text": h["doc"]} for i, h in enumerate(hits)]
)
st.dataframe(hits_df, use_container_width=True)

# Generate answer with OpenAI (optional if key missing)
api_key = safe_get_openai_key()
if not api_key:
    st.warning(
        "No OPENAI_API_KEY found. I can retrieve relevant rows, but I can’t generate an LLM answer.\n\n"
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

st.markdown("---")
st.caption("RAG flow: CSV rows → text chunks → embeddings → FAISS retrieval → LLM answer grounded in retrieved rows.")
