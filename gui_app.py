import json
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Azure Docs Q&A", page_icon="💬", layout="centered")
st.title("💬 Azure Documentation Q&A")
st.caption("Ask a question about Azure and get an answer grounded in your documentation.")

# ---------------------------------------------------------------------------
# Sidebar setup: API key + dataset
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("Gemini API Key", type="password")
    dataset_file = st.file_uploader("Upload dataset.json", type=["json"])
    build_clicked = st.button("Build / Rebuild Index", use_container_width=True)
    st.divider()
    st.caption(
        "dataset.json must contain a list of records with **url**, **title** "
        "and **content** fields."
    )

# ---------------------------------------------------------------------------
# Cached heavy resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def load_data(file) -> pd.DataFrame:
    return pd.read_json(file)


def validate_data(data: pd.DataFrame) -> bool:
    if data.empty:
        return False
    if not all(col in data.columns for col in ["url", "title", "content"]):
        return False
    return True


def clean_text(text: str) -> str:
    return text.strip()


def build_index(file):
    """Level 1: load -> clean -> chunk -> embed -> store in Chroma."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import chromadb
    from rank_bm25 import BM25Okapi

    data = load_data(file)
    if not validate_data(data):
        raise ValueError("dataset.json must have 'url', 'title' and 'content' columns.")

    data["content"] = data["content"].apply(clean_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for _, row in data.iterrows():
        for i, chunk in enumerate(splitter.split_text(row["content"])):
            chunks.append(
                {"text": chunk, "title": row["title"], "url": row["url"], "chunk_id": i}
            )

    embedder = get_embedder()
    embeddings = embedder.encode([c["text"] for c in chunks])

    client = chromadb.Client()  # in-memory, rebuilt each session
    collection = client.get_or_create_collection(name="azure_collection")
    collection.add(
        documents=[c["text"] for c in chunks],
        metadatas=[{"title": c["title"], "url": c["url"], "chunk_id": c["chunk_id"]} for c in chunks],
        embeddings=embeddings.tolist(),
        ids=[str(i) for i in range(len(chunks))],
    )

    documents = [c["text"] for c in chunks]
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    return {"collection": collection, "documents": documents, "bm25": bm25}


# ---------------------------------------------------------------------------
# Level 2: query rewriting + classification
# ---------------------------------------------------------------------------
def rewrite_query(model, query: str) -> str:
    prompt = f"""
Rewrite the following Azure documentation search query.

Rules:
- keep it in question format.
- Keep the same meaning.
- Use Azure terminology.
- Do not answer the question.

Query:
{query}
"""
    response = model.generate_content(prompt)
    return response.text.strip()


def classify_query(model, query: str) -> str:
    prompt = f"""
Classify the following Azure question into ONE category.

Categories:

Compute
Storage
Networking
Identity
Database
AI
Monitoring
Security
General
Out of scope

Question:
{query}

Return only the category.
"""
    response = model.generate_content(prompt)
    return response.text.strip()


# ---------------------------------------------------------------------------
# Level 3: hybrid (dense + sparse) retrieval
# ---------------------------------------------------------------------------
def dense_search(query, collection, k=5):
    return collection.query(query_texts=[query], n_results=k)


def sparse_search(query, bm25, documents, k=5):
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:k]
    return [documents[i] for i in top_indices]


def hybrid_search_weighted(query, collection, bm25, documents, alpha=0.6, k=5):
    dense = dense_search(query, collection, k)["documents"][0]
    sparse = sparse_search(query, bm25, documents, k)

    scores = {}
    for rank, doc in enumerate(dense):
        scores[doc] = scores.get(doc, 0) + alpha * ((k - rank) / k)
    for rank, doc in enumerate(sparse):
        scores[doc] = scores.get(doc, 0) + (1 - alpha) * ((k - rank) / k)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Level 1: final answer generation
# ---------------------------------------------------------------------------
def generate_answer(model, query, context):
    prompt = f"""
You are an AI assistant for Microsoft Azure documentation.

Answer the user's question using ONLY the information provided in the context below.

Rules:
- Do NOT use outside knowledge.
- Do NOT make up information.
- If the answer is not found in the context, reply:
  "I couldn't find the answer in the provided documentation."
- Give a concise and accurate answer.

Context:
{context}

Question:
{query}

Answer:
"""
    response = model.generate_content(prompt)
    return response.text.strip()


# ---------------------------------------------------------------------------
# Build index on demand
# ---------------------------------------------------------------------------
if build_clicked:
    if dataset_file is None:
        st.sidebar.error("Please upload a dataset.json file first.")
    else:
        with st.spinner("Building index... this may take a minute."):
            st.session_state["index"] = build_index(dataset_file)
        st.sidebar.success("Index built successfully!")

index_ready = "index" in st.session_state

# ---------------------------------------------------------------------------
# Main Q&A area
# ---------------------------------------------------------------------------
query = st.text_input("Ask a question about Azure:", placeholder="e.g. How do I deploy a Python web app?")
ask_clicked = st.button("Ask", type="primary", use_container_width=True)

if ask_clicked:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not index_ready:
        st.error("Please upload a dataset.json and click 'Build / Rebuild Index' in the sidebar first.")
    elif not query.strip():
        st.warning("Please type a question.")
    else:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        with st.spinner("Thinking..."):
            idx = st.session_state["index"]

            rewritten = rewrite_query(model, query)
            results = hybrid_search_weighted(
                rewritten, idx["collection"], idx["bm25"], idx["documents"], alpha=0.6, k=5
            )
            context = "\n\n".join(doc for doc, _ in results[:5])
            answer = generate_answer(model, rewritten, context)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Details"):
            st.markdown(f"**Rewritten query:** {rewritten}")
            st.markdown("**Retrieved context chunks:**")
            for i, (doc, score) in enumerate(results[:5], 1):
                st.markdown(f"*Rank {i} (score {score:.2f})*")
                st.text(doc[:300])
