# 🚀 Route Graduation Project: Intelligent RAG System

This repository contains a complete, production-ready Intelligent Retrieval-Augmented Generation (RAG) system built for Microsoft Azure Official Documentation. 

The architecture transitions from basic vector retrieval to a highly advanced Hybrid Search system with self-querying AI routing.

---

## 🛠️ Setup and Run Instructions

### Prerequisites
- Python 3.10+
- A Google Gemini API Key

### 1. Installation
1. Clone this repository to your local machine.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
   - **Windows (CMD):** `.\venv\Scripts\activate.bat`
   - **Mac/Linux:** `source venv/bin/activate`
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Configuration
1. Open the `.env.example` file.
2. Rename it to `.env` (or create a new `.env` file).
3. Paste your Gemini API key inside:
   ```env
   GEMINI_API_KEY="AIzaSyYourActualKeyHere..."
   ```

### 3. Running the Application
The final deployment is a Streamlit Web Application that bypasses slow startup times by connecting directly to the pre-populated `chroma_db` database.

To run the GUI:
```bash
python -m streamlit run gui_app.py
```

---

## 📂 Project Structure & Notebooks

To see the exact logic behind the scenes, you can review our custom Native Python implementations:

1. **`level 1.ipynb` (Foundation RAG):** 
   - Uses `RecursiveCharacterTextSplitter` (Size: 1000, Overlap: 150).
   - Embeds using `all-MiniLM-L6-v2`.
   - Saves to a persistent **ChromaDB** on disk.
2. **`level_2.ipynb` (Query Intelligence):** 
   - Implements Self-Querying.
   - Forces the LLM to Rewrite, Classify, and Extract Metadata `JSON` filters from user queries for 100% precision vector lookups.
3. **`level_3.ipynb` (Hybrid Search Retrieval):** 
   - Combines Dense Search (Semantic/Chroma) and Sparse Search (Keyword/BM25).
   - Fuses results mathematically using **Reciprocal Rank Fusion (RRF)**.
4. **`gui_app.py` (Final Deployment):** 
   - The production Streamlit interface integrating all 3 levels.

---

## 📊 Evaluation & Comparisons

### Example Queries & Outputs
* **Query:** "How do I deploy a web app?"
  * **Rewritten by AI:** "How do I deploy a web application to Azure App Service?"
  * **Classification:** `Compute`
  * **Result:** Successfully generates a grounded answer based exclusively on Azure documentation context.

### Dense-Only vs. Hybrid Search Comparison
* **The Problem with Dense Only:** Semantic search understands meaning but often fails to find exact jargon (e.g., "Error Code 404").
* **The Problem with Sparse Only:** BM25 finds exact strings but doesn't understand synonyms.
* **The Hybrid Solution:** By using Reciprocal Rank Fusion (RRF), our system queries both ChromaDB and BM25 simultaneously. We assign a mathematical weight of `0.6` to Dense and `0.4` to Sparse, guaranteeing that retrieved documents match both the *meaning* of the question and the *exact technical keywords* used.

---

## 📝 Written Reports
Please refer to the following Markdown reports included in this submission:
1. `final_project_report.md` - Complete technical overview.
2. `design_decisions_report.md` - Why native Python was chosen over LangChain.