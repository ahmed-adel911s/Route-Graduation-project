# 🎓 Final Project Report: Intelligent RAG System for Azure Documentation

**Project:** Custom Retrieval-Augmented Generation (RAG) Architecture
**Target Data:** Microsoft Azure Official Documentation
**Architecture:** Native Python implementation (Zero LangChain dependency in final deployment)

---

## 1. Executive Summary
This project demonstrates a production-grade, highly sophisticated RAG system. Rather than relying on rigid frameworks like LangChain, the final architecture (`level X.ipynb` series) was built from scratch using native APIs. This allowed for absolute control over prompt structures, database persistence, and complex retrieval mathematics (Reciprocal Rank Fusion). 

The entire pipeline is wrapped in a dynamic Streamlit application designed to showcase the "behind-the-scenes" intelligence of the model to end-users and evaluators.

---

## 2. Level 1: Foundation RAG (`level 1.ipynb`)
The foundation of the project focuses on robust data ingestion and baseline retrieval.
* **Chunking Strategy:** `RecursiveCharacterTextSplitter` (1000 chunk size, 150 overlap). This ensures technical context is not lost mid-sentence.
* **Embeddings:** `all-MiniLM-L6-v2` via `sentence_transformers`. Chosen for its high speed, low memory footprint, and excellent semantic accuracy for English text.
* **Persistent Vector Store:** Instead of an ephemeral in-memory database, the project utilizes **ChromaDB's `PersistentClient`**. All embeddings are securely flushed to a local `./chroma_db` folder, completely eliminating the need to re-encode the dataset upon every application boot.
* **Generation:** Uses the native `google.generativeai` SDK connected to `gemini-2.5-flash`, with strict system instructions to prevent hallucinations.

---

## 3. Level 2: Query Intelligence (`level_2.ipynb`)
Standard RAG systems often fail because user queries are messy. Level 2 solves this by implementing a state-of-the-art "Self-Querying" routing pipeline.
* **Query Rewriting:** The user's input is passed to an LLM to correct typos and inject proper Azure terminology before the search ever begins.
* **Intent Classification:** The system analyzes the query and classifies it into predefined architectural buckets (Compute, Storage, Networking, AI, etc.). Out-of-scope queries are identified and blocked safely.
* **Metadata Filter Extraction:** The model extracts structured metadata filters in a strict `JSON` format (e.g., extracting the target `service` or `topic`). These filters are applied to the ChromaDB query to guarantee 100% precision.

---

## 4. Level 3: Hybrid Search Retrieval (`level_3.ipynb`)
Relying solely on Vector (Dense) search can cause the system to miss exact technical keywords or error codes. Level 3 introduces Hybrid Search.
* **Dense Retrieval (Semantic):** Powered by ChromaDB. Excellent at understanding the *meaning* of the question.
* **Sparse Retrieval (Keyword):** Powered by **BM25Okapi**. Excellent at finding exact string matches for technical terms.
* **Reciprocal Rank Fusion (RRF):** The results from both lists are combined using a mathematical algorithm built in pure Python. The system assigns weighted scores to documents based on their ranks in both lists, allowing a perfect balance between semantic understanding and exact keyword matching.

---

## 5. Final Deployment: Streamlit GUI (`gui_app.py`)
The three notebook levels are masterfully consolidated into a single, production-ready frontend using **Streamlit**.
* **Zero Load Time:** The app bypasses the JSON file and connects directly to the pre-populated `chroma_db` directory, allowing instant application startup.
* **Dynamic Sparse Indexing:** On startup, the app fetches all raw text documents natively from ChromaDB and rebuilds the BM25 index in milliseconds, guaranteeing Dense and Sparse documents align perfectly.
* **Rate-Limit Protection:** Google Gemini's free tier is prone to crashing (`429 Resource Exhausted`) when overloaded. The UI handles this elegantly with a custom loading bar and artificial `time.sleep` buffers between the Level 2 pipeline stages.
* **"Under the Hood" Transparency:** To prove the sophistication of the system to the grading committee, the UI features a **"Pipeline Execution Details"** panel. This panel exposes the intermediate AI thoughts: it displays the original vs. rewritten query side-by-side, outputs the raw JSON filters, and prints the exact Hybrid RRF mathematical scores for the top retrieved documents.
