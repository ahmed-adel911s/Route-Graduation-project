import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import chromadb
from rank_bm25 import BM25Okapi
import google.generativeai as genai

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Azure Docs Q&A", page_icon="💬", layout="wide")
st.title("💬 Azure Documentation Q&A (Intelligent RAG)")
st.caption("Graduation Project: Combining Foundation RAG, Query Intelligence, and Hybrid Search")

# ---------------------------------------------------------------------------
# Sidebar setup: API key
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.divider()
    st.markdown("""
    **Project Levels Implemented:**
    - ✅ Level 1: Foundation RAG (ChromaDB)
    - ✅ Level 2: Query Intelligence (Self-Querying)
    - ✅ Level 3: Hybrid Search (Dense + Sparse)
    """)

# ---------------------------------------------------------------------------
# Cached heavy resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_databases():
    """Connect to Persistent ChromaDB and rebuild BM25 index from its contents."""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("azure_collection")
        
        # Fetch all documents to build BM25 Sparse Index
        docs_data = collection.get()
        documents = docs_data.get("documents", [])
        
        if not documents:
            return None
            
        tokenized_docs = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        
        return {
            "collection": collection, 
            "documents": documents, 
            "bm25": bm25
        }
    except Exception as e:
        return None

# Load the indices on startup
idx = load_databases()
if idx is None:
    st.error("❌ Failed to load `./chroma_db`. Please run `level 1.ipynb` first to build the database.")
    st.stop()

# ---------------------------------------------------------------------------
# Level 2: Query Intelligence
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
    return model.generate_content(prompt).text.strip()

def classify_query(model, query: str) -> str:
    prompt = f"""
Classify the following Azure question into ONE category.
Categories:
Compute, Storage, Networking, Identity, Database, AI, Monitoring, Security, General, Out of scope

Question:
{query}

Return only the category.
"""
    return model.generate_content(prompt).text.strip()

def extract_filters(model, query: str) -> dict:
    prompt = f"""
Extract the following information from the Azure documentation query.
Return EXACTLY a valid JSON object with these keys:
- category
- service
- language
- topic

Query:
{query}
"""
    response_text = model.generate_content(prompt).text.strip()
    # Clean the markdown JSON wrapper if it exists
    if response_text.startswith("```json"):
        response_text = response_text[7:-3].strip()
    elif response_text.startswith("```"):
        response_text = response_text[3:-3].strip()
        
    try:
        return json.loads(response_text)
    except:
        return {"error": "Failed to parse JSON"}

# ---------------------------------------------------------------------------
# Level 3: Hybrid (Dense + Sparse) Retrieval
# ---------------------------------------------------------------------------
def dense_search(query, collection, k=5):
    return collection.query(query_texts=[query], n_results=k)["documents"][0]

def sparse_search(query, bm25, documents, k=5):
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:k]
    return [documents[i] for i in top_indices]

def hybrid_search_weighted(query, collection, bm25, documents, alpha=0.6, k=5):
    dense = dense_search(query, collection, k)
    sparse = sparse_search(query, bm25, documents, k)

    scores = {}
    for rank, doc in enumerate(dense):
        scores[doc] = scores.get(doc, 0) + alpha * ((k - rank) / k)
    for rank, doc in enumerate(sparse):
        scores[doc] = scores.get(doc, 0) + (1 - alpha) * ((k - rank) / k)

    # Return tuples of (document, score)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# ---------------------------------------------------------------------------
# Level 1: Final Answer Generation
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
    return model.generate_content(prompt).text.strip()


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.markdown("### Ask a Question")
query = st.text_input("Enter your Azure query here:", placeholder="e.g. How can I deploy a Java web application to Azure App Service?", label_visibility="collapsed")
ask_clicked = st.button("Generate Answer", type="primary")

if ask_clicked:
    if not api_key:
        st.error("⚠️ Please enter your Gemini API key in the sidebar.")
    elif not query.strip():
        st.warning("⚠️ Please type a question.")
    else:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        # Initialize placeholders for smooth UI loading
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # Step 1: Rewrite
            status_text.info("🔄 Rewriting query using Azure terminology...")
            rewritten_query = rewrite_query(model, query)
            progress_bar.progress(20)
            time.sleep(4) # Prevent Rate Limits

            # Step 2: Classify
            status_text.info("🗂️ Classifying intent...")
            classification = classify_query(model, rewritten_query)
            progress_bar.progress(40)
            time.sleep(4) # Prevent Rate Limits

            # Step 3: Extract Filters
            status_text.info("🔎 Extracting metadata filters...")
            filters_json = extract_filters(model, rewritten_query)
            progress_bar.progress(60)
            time.sleep(4) # Prevent Rate Limits

            # Step 4: Hybrid Search
            status_text.info("⚖️ Performing Hybrid Search (Dense + Sparse)...")
            results = hybrid_search_weighted(
                rewritten_query, 
                idx["collection"], 
                idx["bm25"], 
                idx["documents"], 
                alpha=0.6, 
                k=5
            )
            context = "\n\n".join(doc for doc, _ in results[:5])
            progress_bar.progress(80)

            # Step 5: Generate Answer
            status_text.info("🤖 Generating final grounded answer...")
            time.sleep(4) # Prevent Rate Limits
            final_answer = generate_answer(model, rewritten_query, context)
            
            # Clear loading elements
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()

            # Display Final Answer
            st.success("✅ **Answer Generated Successfully!**")
            st.write(final_answer)

            # Display Execution Details (Grading/Rubric proof)
            with st.expander("🔍 Pipeline Execution Details (Levels 1, 2, 3)"):
                st.markdown("### 🧠 Level 2: Query Intelligence")
                col1, col2, col3 = st.columns(3)
                col1.metric("Original Query", query)
                col1.metric("Rewritten Query", rewritten_query)
                col2.metric("Classification", classification)
                with col3:
                    st.write("**Extracted Filters:**")
                    st.json(filters_json)
                
                st.divider()
                st.markdown("### ⚖️ Level 3: Hybrid Search Retrieval")
                st.markdown(f"**Top Retrieved Contexts (Weighted RRF Score):**")
                for i, (doc, score) in enumerate(results[:5], 1):
                    st.info(f"**Rank {i} | Score: {score:.3f}**\n\n{doc[:300]}...")

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.warning("If you got a 429 Error, the API rate limit was hit. Try waiting a minute and asking again.")
