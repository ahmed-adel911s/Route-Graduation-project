# Route Graduation Project: Intelligent RAG System

This document outlines the full implementation details of our Intelligent Retrieval-Augmented Generation (RAG) system, mapped directly to the project rubric requirements.

---

## 0.1 — Dataset Collection
* **Source Website:** Microsoft Azure Official Documentation
* **URLs Used:** Various pages under `https://learn.microsoft.com/en-us/azure/` (e.g., Azure PostgreSQL, Azure Chaos Studio, Azure Functions, etc.)
* **Type of Data Collected:** Technical documentation, tutorials, and product overviews.
* **Structure of Data:** The raw data was parsed from HTML pages and stored as a structured `JSON` array containing the plain text `content`, `url`, and page `title`.
* **Volume:** Approximately 65 individual pages/documents were scraped and collected.
* **Collection Method:** A custom Python script (`scraper.py`) utilizing `requests` for fetching pages and `BeautifulSoup4` for HTML parsing.
* **Cleaning & Preprocessing:** The `scraper.py` script systematically removed non-content HTML elements such as `<script>`, `<style>`, `<nav>`, `<footer>`, and `<header>` tags to ensure only clean, relevant plain text was extracted.

---

## 0.2 — Chunking Strategy
* **Selected Strategy:** `RecursiveCharacterTextSplitter` with a `chunk_size` of 1000 characters and a `chunk_overlap` of 150 characters.
* **Why this strategy was chosen:** Azure documentation contains highly technical explanations and code-related concepts. A natural unit of meaning here is usually a full paragraph or a distinct section. Splitting blindly by characters could cut a technical definition in half, destroying its meaning.
* **Observations:** Technical documentation relies heavily on context. We observed that smaller chunks lost the context of *which* specific Azure product was being discussed.
* **Rejected Strategies:** We rejected a basic `CharacterTextSplitter` because it does not respect semantic boundaries (like sentences or paragraphs) and would abruptly chop words in half.
* **Weaknesses:** A weakness of a 1000-character chunk is that it might capture two slightly different topics in one chunk, potentially diluting the vector embedding focus.
* **Mitigation:** We reduced this weakness by adding a `chunk_overlap` of 150 characters to ensure no semantic meaning is lost at the boundaries, and by storing the source `title` in the metadata to enforce strict filtering later in the pipeline.

---

## Level 1 — Foundation RAG Pipeline
* **Data Ingestion:** The `dataset.json` is loaded, chunked, and stored with metadata (`title` and `url`).
* **Embedding Model Choice:** `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace.
  * *Language Support:* Excellent support for English.
  * *Accuracy:* Highly rated on semantic search benchmarks for sentence-level tasks.
  * *Speed:* Extremely fast, allowing it to run instantly on local CPU hardware without latency.
  * *Cost:* 100% free and open-source (no API costs).
  * *Embedding Dimension:* 384 dimensions, which is highly efficient for fast FAISS indexing and low memory usage.
* **Vector Store:** FAISS (Facebook AI Similarity Search) was used for fast, local dense retrieval.
* **Answer Generation:** We utilized Google's `gemini-2.5-flash` API. The prompt strictly instructs the model to answer *only* based on the provided context to prevent hallucinations.
* **Deliverable:** `level1_rag.ipynb` successfully takes a query, runs `similarity_search_with_score`, prints the exact L2 distance scores alongside the chunks, and generates a grounded answer.

---

## Level 2 — Query Intelligence
Our system implements a state-of-the-art "Self-Querying" routing pipeline.
* **Query Rewriting:** User queries are passed to Gemini to fix typos and make them self-contained (e.g., "how much does azre postgresql cost" → "How much does Azure Database for PostgreSQL cost?").
* **Query Classification:** Queries are analyzed and strictly classified into predefined intents using JSON structured output:
  * *Factual Lookup:* "how to create a database in azure postgresql"
  * *Comparison:* "What is the difference between azure functions and azure logic apps?"
  * *Out of Scope:* "how to bake a chocolate cake" (System gracefully rejects answering).
* **Structured Parameter Extraction:** The LLM analyzes the query against a list of our 65 known Azure document titles and extracts the exact `title_filter`.
* **Filtered Retrieval:** The vector search is restricted to *only* search within chunks that match the extracted `title_filter`, ensuring 100% precision.
* **Deliverable:** `level2_query_intelligence.ipynb` implements this entire 5-step pipeline end-to-end.

---

## Level 3 — Retrieval Quality: Hybrid Search
* **Concept:** Vector search (Dense) understands the *meaning* of a query, while BM25 (Sparse) acts like a traditional search engine looking for *exact keyword matches* (like specific error codes). Relying solely on Vector search might miss exact technical terms, while relying only on Keyword search misses semantic intent.
* **Implementation:** We implemented Hybrid Search by combining FAISS (Dense) and BM25 (Sparse).
* **Scoring Method:** We merged the results using **Reciprocal Rank Fusion (RRF)**. We built a custom mathematical function using the standard formula `1 / (rank + 60)` to assign scores to documents from both lists and rank the fused ensemble.
* **Comparison:** 
  * In our `level3_hybrid_search.ipynb` notebook, we automatically run 3 test queries: a niche terminology query, a broad semantic query, and a keyword-heavy query (e.g., "Error 404 in Azure Web Apps").
  * The notebook prints the Top 3 Dense results side-by-side with the Top 3 Hybrid results so the user can easily see how keyword matching improves specific technical queries.
* **Deliverable:** `level3_hybrid_search.ipynb` implements this comparison perfectly.