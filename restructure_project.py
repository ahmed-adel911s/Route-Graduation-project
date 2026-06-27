import json
import os
import nbformat as nbf

# --- FIX LEVEL 1 ---
with open('level1_rag.ipynb', 'r', encoding='utf-8') as f:
    d = json.load(f)

# Cell 7 source replacement
new_cell_7 = """# Load the vector store (optional, if running in a new session)
# vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

query = "What is Azure Chaos Studio?"
print(f"Query: {query}\\n")

# Retrieve top 3 most similar chunks WITH SCORES (L2 Distance: lower is better)
docs_and_scores = vector_store.similarity_search_with_score(query, k=3)

# Extract just the docs for the next chain
retrieved_docs = [doc for doc, score in docs_and_scores]

for i, (doc, score) in enumerate(docs_and_scores):
    print(f"--- Chunk {i+1} ---")
    print(f"L2 Distance Score: {score:.4f} (Lower = More Similar)")
    print(f"Metadata: {doc.metadata}")
    print(f"Content snippet: {doc.page_content[:200]}...\\n")"""

d['cells'][7]['source'] = [line + '\n' for line in new_cell_7.split('\n')]
d['cells'][7]['source'][-1] = d['cells'][7]['source'][-1].rstrip('\n')

with open('level1_rag.ipynb', 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=1)


# --- CREATE LEVEL 2 ---
nb2 = nbf.v4.new_notebook()

text2_1 = """# Level 2: Query Intelligence

This notebook implements the advanced routing pipeline required for Level 2.
Pipeline steps: Original query -> Query rewriting -> Query classification -> Filter extraction -> Filtered retrieval -> Answer generation"""

code2_1 = """import os
import json
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
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
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.1)

# Load dataset and chunk it
with open('dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

docs = []
for item in dataset:
    doc = Document(page_content=item['content'], metadata={"title": item['title'], "url": item['url']})
    docs.append(doc)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
all_chunks = text_splitter.split_documents(docs)
available_titles = list(set([doc.metadata['title'] for doc in all_chunks]))"""

text2_2 = """## Query Rewriting, Classification & Filter Extraction"""

code2_2 = """# Prompt for Query Analysis
analyzer_template = \"\"\"You are an expert AI search analyzer. Analyze the user's query about Azure.

Available Azure Product Titles:
{titles}

Your tasks:
1. "rewritten_query": Rewrite the query to be a highly descriptive, self-contained search query. Fix typos.
2. "intent": Classify the intent strictly as one of: [factual_lookup, comparison, recommendation, out_of_scope]
3. "title_filter": If the user is specifically asking about a product in the "Available Azure Product Titles" list, output the exact title string. If they are asking a general question, output null.

Return ONLY a valid JSON object. Do not include markdown formatting.
{{
  "rewritten_query": "...",
  "intent": "...",
  "title_filter": "..."
}}

Original Query: {query}
\"\"\"

analyzer_prompt = PromptTemplate.from_template(analyzer_template)

def analyze_query(query):
    titles_str = "\\n".join([f"- {t}" for t in available_titles])
    chain = analyzer_prompt | llm | StrOutputParser()
    result = chain.invoke({"titles": titles_str, "query": query})
    result = result.replace("```json", "").replace("```", "").strip()
    return json.loads(result)

# Test 3 different query categories for the rubric!
test_queries = [
    "What is the difrence between azure functions and azure logic apps?", # Comparison
    "how to create a database in azure postgresql", # Factual lookup
    "how to bake a chocolate cake" # Out of scope
]

for q in test_queries:
    print(f"\\n--- Raw Query: {q} ---")
    analysis = analyze_query(q)
    print(json.dumps(analysis, indent=2))"""

text2_3 = """## Filtered Retrieval & Answer Generation"""

code2_3 = """def process_end_to_end(raw_query):
    print(f"\\n==========================\\nProcessing: {raw_query}")
    analysis = analyze_query(raw_query)
    
    if analysis['intent'] == 'out_of_scope':
        return "I am an Azure assistant. I cannot answer questions outside of Azure documentation."
        
    query_to_search = analysis['rewritten_query']
    title_filter = analysis['title_filter']
    
    # 4. Filtered Retrieval
    if title_filter:
        filtered_chunks = [c for c in all_chunks if c.metadata['title'] == title_filter]
    else:
        filtered_chunks = all_chunks
        
    if not filtered_chunks:
        return "No relevant documents found for filtering."
        
    temp_vector_store = FAISS.from_documents(filtered_chunks, embeddings)
    retrieved_docs = temp_vector_store.similarity_search(query_to_search, k=3)
    
    # 5. Answer Generation
    answer_template = \"\"\"You are an expert AI assistant.
Use the following extremely relevant, filtered context to answer the question.
If you don't know the answer, say you don't know. Do NOT hallucinate.

Context:
{context}

Question: {question}

Answer:\"\"\"

    answer_prompt = PromptTemplate.from_template(answer_template)
    context_str = "\\n\\n".join([doc.page_content for doc in retrieved_docs])
    
    final_chain = answer_prompt | llm | StrOutputParser()
    return final_chain.invoke({
        "context": context_str, 
        "question": query_to_search
    })

# Test the factual query end-to-end
final_answer = process_end_to_end("how to create a database in azure postgresql")
print(f"\\nFINAL ANSWER:\\n{final_answer}")"""

nb2['cells'] = [
    nbf.v4.new_markdown_cell(text2_1), nbf.v4.new_code_cell(code2_1),
    nbf.v4.new_markdown_cell(text2_2), nbf.v4.new_code_cell(code2_2),
    nbf.v4.new_markdown_cell(text2_3), nbf.v4.new_code_cell(code2_3)
]
with open('level2_query_intelligence.ipynb', 'w') as f:
    nbf.write(nb2, f)


# --- CREATE LEVEL 3 ---
nb3 = nbf.v4.new_notebook()

text3_1 = """# Level 3: Retrieval Quality (Hybrid Search)

This notebook compares Dense-Only retrieval (FAISS) against Hybrid Search (FAISS + BM25) across 3 different queries."""

code3_1 = """import os
import json
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Initialize models
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

with open('dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

docs = []
for item in dataset:
    doc = Document(page_content=item['content'], metadata={"title": item['title'], "url": item['url']})
    docs.append(doc)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
all_chunks = text_splitter.split_documents(docs)

# Build Dense Retriever (FAISS)
vector_store = FAISS.from_documents(all_chunks, embeddings)
dense_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Build Sparse Retriever (BM25)
sparse_retriever = BM25Retriever.from_documents(all_chunks)
sparse_retriever.k = 5"""

text3_2 = """## Implement Reciprocal Rank Fusion (Hybrid)"""

code3_2 = """def hybrid_search(query, k=5):
    dense_results = dense_retriever.invoke(query)
    sparse_results = sparse_retriever.invoke(query)
    
    rrf_scores = {}
    for rank, doc in enumerate(dense_results):
        doc_content = doc.page_content
        if doc_content not in rrf_scores:
            rrf_scores[doc_content] = {"score": 0.0, "doc": doc}
        rrf_scores[doc_content]["score"] += 1.0 / (rank + 60)
            
    for rank, doc in enumerate(sparse_results):
        doc_content = doc.page_content
        if doc_content not in rrf_scores:
            rrf_scores[doc_content] = {"score": 0.0, "doc": doc}
        rrf_scores[doc_content]["score"] += 1.0 / (rank + 60)
        
    sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_docs][:k]"""

text3_3 = """## Comparison: Dense vs Hybrid Search"""

code3_3 = """test_queries = [
    "What is Azure Chaos Studio?", # Niche/Specific terminology
    "How do I manage relational databases?", # Broad semantic query
    "Error 404 in Azure Web Apps" # Keyword-heavy query
]

for q in test_queries:
    print(f"\\n=======================================================")
    print(f"QUERY: {q}")
    print(f"=======================================================\\n")
    
    print("--- TOP 3 DENSE RESULTS (FAISS) ---")
    dense_docs = dense_retriever.invoke(q)[:3]
    for i, doc in enumerate(dense_docs):
        print(f"{i+1}. [{doc.metadata['title']}] {doc.page_content[:100]}...")
        
    print("\\n--- TOP 3 HYBRID RESULTS (FAISS + BM25) ---")
    hybrid_docs = hybrid_search(q, k=3)
    for i, doc in enumerate(hybrid_docs):
        print(f"{i+1}. [{doc.metadata['title']}] {doc.page_content[:100]}...")
        
    print("\\n\\n")"""

nb3['cells'] = [
    nbf.v4.new_markdown_cell(text3_1), nbf.v4.new_code_cell(code3_1),
    nbf.v4.new_markdown_cell(text3_2), nbf.v4.new_code_cell(code3_2),
    nbf.v4.new_markdown_cell(text3_3), nbf.v4.new_code_cell(code3_3)
]
with open('level3_hybrid_search.ipynb', 'w') as f:
    nbf.write(nb3, f)

# Delete old files
old_files = [
    "level2_rag_advanced.ipynb", 
    "level3_rag_pipeline.ipynb", 
    "create_level2_notebook.py", 
    "create_level3_notebook.py",
    "fix_level2_notebook.py",
    "fix_level3_notebook.py",
    "fix_model.py"
]
for file in old_files:
    if os.path.exists(file):
        os.remove(file)
