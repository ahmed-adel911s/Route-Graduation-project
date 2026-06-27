import nbformat as nbf
import json

with open('level3_rag_pipeline.ipynb', 'r', encoding='utf-8') as f:
    d = json.load(f)

# The cells:
# cell 0: markdown
# cell 1: imports and setup
# cell 2: markdown
# cell 3: chunks
# cell 4: markdown
# cell 5: rewrite prompt
# cell 6: markdown
# cell 7: hybrid search function

# Replace cell 1
new_code1 = """import os
import json
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Setup & API Keys
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize models
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.1)"""

# Replace cell 7 (Filtered Hybrid Retrieval)
new_code7 = """def reciprocal_rank_fusion(retriever_results, k=60):
    \"\"\"
    Fuses multiple ranked lists of documents using Reciprocal Rank Fusion (RRF).
    \"\"\"
    rrf_scores = {}
    for doc_list in retriever_results:
        for rank, doc in enumerate(doc_list):
            doc_content = doc.page_content
            if doc_content not in rrf_scores:
                rrf_scores[doc_content] = {"score": 0.0, "doc": doc}
            # RRF formula: 1 / (rank + k)
            rrf_scores[doc_content]["score"] += 1.0 / (rank + k)
            
    # Sort by score descending
    sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_docs]

def hybrid_filtered_search(analysis_dict, k=3):
    query_to_search = analysis_dict['rewritten_query']
    title_filter = analysis_dict['title_filter']
    
    # Apply the dynamic filter!
    if title_filter:
        print(f"\\n[*] Applying strict filter for: {title_filter}")
        filtered_chunks = [c for c in all_chunks if c.metadata['title'] == title_filter]
    else:
        print("\\n[*] No specific product filter detected. Searching globally.")
        filtered_chunks = all_chunks
        
    if not filtered_chunks:
        return []
        
    print(f"[*] Search space reduced to {len(filtered_chunks)} chunks.")
    
    # 1. FAISS Vector Search (Dense)
    temp_vector_store = FAISS.from_documents(filtered_chunks, embeddings)
    faiss_retriever = temp_vector_store.as_retriever(search_kwargs={"k": k})
    faiss_docs = faiss_retriever.invoke(query_to_search)
    print(f"[*] FAISS Vector Search found {len(faiss_docs)} matches.")
    
    # 2. BM25 Keyword Search (Sparse)
    bm25_retriever = BM25Retriever.from_documents(filtered_chunks)
    bm25_retriever.k = k
    bm25_docs = bm25_retriever.invoke(query_to_search)
    print(f"[*] BM25 Keyword Search found {len(bm25_docs)} matches.")
    
    # 3. Ensemble (Hybrid Search) via Custom Reciprocal Rank Fusion
    final_hybrid_docs = reciprocal_rank_fusion([faiss_docs, bm25_docs])
    
    # Return top K final docs
    return final_hybrid_docs[:k]

retrieved_docs = hybrid_filtered_search(analysis)
print(f"\\n[*] Successfully retrieved {len(retrieved_docs)} optimized hybrid chunks.")"""

d['cells'][1]['source'] = [line + '\n' for line in new_code1.split('\n')]
d['cells'][7]['source'] = [line + '\n' for line in new_code7.split('\n')]

# Clean up trailing newlines
d['cells'][1]['source'][-1] = d['cells'][1]['source'][-1].rstrip('\\n')
d['cells'][7]['source'][-1] = d['cells'][7]['source'][-1].rstrip('\\n')

with open('level3_rag_pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=1)
