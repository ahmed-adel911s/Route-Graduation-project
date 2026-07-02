# 🏛️ Architectural Design Decisions Report
**Project:** Intelligent Retrieval-Augmented Generation (RAG) System

This document outlines the core technical design decisions made during the development of the Azure Documentation RAG system, explaining the *why* behind the architecture.

---

## 1. Dropping LangChain for Native "Clean" Architecture
**Decision:** We transitioned from the initial LangChain prototypes (`levelX_name.ipynb`) to a native Python implementation (`level X.ipynb`).
**Rationale:**
* **Control:** Frameworks like LangChain abstract away the core mechanics of RAG (like prompt formatting and retrieval logic). By writing the system using native `google.generativeai` and `chromadb` APIs, we gained 100% control over the exact payloads being sent and received.
* **Reduced Bloat:** LangChain chains can be difficult to debug when something goes wrong. Native implementation is significantly lighter and easier to maintain in a production environment.
* **Demonstrated Mastery:** Writing algorithms like Reciprocal Rank Fusion mathematically from scratch proves a deeper understanding of the system than simply importing a pre-built framework function.

## 2. Choosing ChromaDB over FAISS
**Decision:** We replaced FAISS with ChromaDB (`PersistentClient`) for the final deployment.
**Rationale:**
* **Persistence:** FAISS is primarily an in-memory index. While it can be saved, ChromaDB operates as a true local database. By pointing to `./chroma_db`, our Streamlit app achieves **zero-latency startup times** because the embeddings are permanently stored on disk.
* **Metadata Filtering:** ChromaDB provides superior, SQL-like syntax for filtering vectors based on metadata (e.g., `collection.query(where={"title": "Azure Functions"})`), which was mandatory for our Level 2 Query Intelligence pipeline.

## 3. The Chunking Strategy
**Decision:** We used the `RecursiveCharacterTextSplitter` with a `chunk_size` of 1000 characters and a `chunk_overlap` of 150 characters.
**Rationale:**
* **Semantic Boundaries:** Azure documentation is highly technical. A blind character splitter might cut a sentence explaining a critical architecture concept right down the middle. The Recursive splitter respects paragraph and sentence boundaries (`\n\n`, `.`).
* **Overlap:** A 150-character overlap ensures that if a technical concept spans across two chunks, the context is carried over, preventing "orphan" information.

## 4. Embedding Model Choice
**Decision:** We selected `sentence-transformers/all-MiniLM-L6-v2`.
**Rationale:**
* **Efficiency:** With only 384 dimensions, the resulting vectors require extremely low memory.
* **Local Execution:** It runs 100% locally on CPU hardware without any API costs, network latency, or rate limits.
* **Accuracy:** Despite its small size, it performs exceptionally well on English semantic similarity benchmarks.

## 5. Implementing Hybrid Search (RRF) Mathematically
**Decision:** We implemented Dense Search (ChromaDB) and Sparse Search (BM25) and fused them using a custom Reciprocal Rank Fusion (RRF) algorithm.
**Rationale:**
* **The Problem:** Semantic search understands *meaning* but struggles with exact technical jargon (e.g., "Error Code 404"). Keyword search finds exact jargon but misses semantic intent.
* **The Solution:** Hybrid search gives us the best of both worlds. By writing the RRF math ourselves, we could introduce a custom `Alpha` parameter (e.g., `0.6` Dense, `0.4` Sparse) to perfectly tune how much weight keyword matching should have against semantic meaning.

## 6. Query Intelligence (Self-Querying Routing)
**Decision:** We force the LLM to rewrite and classify the user's query *before* searching the database.
**Rationale:**
* Users often type sloppy, incomplete, or grammatically incorrect questions.
* If a user searches "how much does it cost?", a vector database has no idea what "it" refers to. By forcing the LLM to rewrite the query into Azure terminology ("How much does Azure Blob Storage cost?"), the semantic similarity scores in the vector search increase dramatically, ensuring near-perfect retrieval accuracy.

## 7. Streamlit UI Buffers (Rate Limit Protection)
**Decision:** We implemented deliberate `time.sleep(4)` pauses between the AI steps in the Streamlit GUI.
**Rationale:**
* **Resilience:** The Google Gemini Free Tier enforces strict `429 Resource Exhausted` rate limits. Because our pipeline makes 4 consecutive API calls (Rewrite → Classify → Extract → Generate), firing them instantly would crash the app.
* **User Experience:** We masked these necessary delays using a sleek progress bar and status text updates in the UI. This protects the application from crashing while visually proving to the user that sophisticated "thinking" is happening under the hood.
