import nbformat as nbf
import json

nb = nbf.v4.new_notebook()

text1 = """# Level 3: Advanced RAG Pipeline (Self-Querying & Hybrid Search)

This notebook implements a state-of-the-art **5-step RAG Pipeline**:
1. **Original query**
2. **Query rewriting:** LLM optimizes the query.
3. **Query classification & Filter extraction:** LLM identifies the specific Azure product and extracts its title as a filter.
4. **Filtered retrieval (Hybrid):** We dynamically filter the chunks and use FAISS (Vector) + BM25 (Keyword) on the subset.
5. **Answer generation:** Final response using the filtered hybrid context."""

code1 = """import os
import json
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
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

text2 = """## Step 1: Reconstruct the Database & Chunks

Because we are doing dynamic metadata filtering, it's highly efficient to keep our raw chunks in memory so we can filter them on the fly before running the Hybrid Search."""

code2 = """# Load dataset and chunk it (takes < 1 second)
with open('dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

docs = []
for item in dataset:
    doc = Document(page_content=item['content'], metadata={"title": item['title'], "url": item['url']})
    docs.append(doc)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
all_chunks = text_splitter.split_documents(docs)

# Extract unique titles for our LLM filter extraction later
available_titles = list(set([doc.metadata['title'] for doc in all_chunks]))
print(f"Loaded {len(all_chunks)} chunks across {len(available_titles)} unique Azure products.")"""

text3 = """## Step 2 & 3: Query Rewriting, Classification & Filter Extraction

We will create a powerful prompt that takes the raw user query, rewrites it for better searchability, determines the intent, and tries to map it to a specific Azure product title."""

code3 = """# Prompt for Query Analysis
analyzer_template = \"\"\"You are an expert AI search analyzer. Analyze the user's query about Azure.

Available Azure Product Titles:
{titles}

Your tasks:
1. "rewritten_query": Rewrite the query to be a highly descriptive, self-contained search query. Fix typos.
2. "intent": Classify the intent as one of: [coding, pricing, architecture, general_info]
3. "title_filter": If the user is specifically asking about a product in the "Available Azure Product Titles" list, output the exact title string. If they are asking a general question across multiple products, output null.

Return ONLY a valid JSON object. Do not include markdown formatting like ```json.
{{
  "rewritten_query": "...",
  "intent": "...",
  "title_filter": "..."
}}

Original Query: {query}
\"\"\"

analyzer_prompt = PromptTemplate.from_template(analyzer_template)

# Create an LLM chain that automatically parses the JSON output
def analyze_query(query):
    titles_str = "\\n".join([f"- {t}" for t in available_titles])
    chain = analyzer_prompt | llm | StrOutputParser()
    result = chain.invoke({"titles": titles_str, "query": query})
    
    # Clean up output in case the LLM adds markdown
    result = result.replace("```json", "").replace("```", "").strip()
    return json.loads(result)

# Test it!
raw_query = "how much does azre postgresql cost"
print(f"Raw Query: {raw_query}")
analysis = analyze_query(raw_query)
print("\\n--- Query Analysis ---")
print(json.dumps(analysis, indent=2))"""

text4 = """## Step 4: Filtered Hybrid Retrieval

Now we use the extracted `title_filter` to filter our chunks. Then we instantly build a temporary FAISS and BM25 index using ONLY the relevant chunks! This guarantees 100% precision with no noise from other Azure products."""

code4 = """def hybrid_filtered_search(analysis_dict, k=3):
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
    
    # 2. BM25 Keyword Search (Sparse)
    bm25_retriever = BM25Retriever.from_documents(filtered_chunks)
    bm25_retriever.k = k
    
    # 3. Ensemble (Hybrid Search) combining both with equal weight
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )
    
    # Execute search
    results = ensemble_retriever.invoke(query_to_search)
    return results

retrieved_docs = hybrid_filtered_search(analysis)
print(f"\\n[*] Retrieved {len(retrieved_docs)} optimized hybrid chunks.")"""

text5 = """## Step 5: Answer Generation

Finally, we pass the strictly filtered, hybrid-ranked chunks to Gemini to generate the final response."""

code5 = """answer_template = \"\"\"You are an expert AI assistant.
Use the following extremely relevant, filtered context to answer the question.
If you don't know the answer, say you don't know. Do NOT hallucinate.

Context:
{context}

Question: {question}

Answer:\"\"\"

answer_prompt = PromptTemplate.from_template(answer_template)

# Format docs into string
context_str = "\\n\\n".join([doc.page_content for doc in retrieved_docs])

# Generate Final Answer
final_chain = answer_prompt | llm | StrOutputParser()
final_answer = final_chain.invoke({
    "context": context_str, 
    "question": analysis['rewritten_query']
})

print("\\n=== FINAL GENERATED ANSWER ===\\n")
print(final_answer)"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text1),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_markdown_cell(text2),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(text3),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_markdown_cell(text4),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_markdown_cell(text5),
    nbf.v4.new_code_cell(code5)
]

with open('level3_rag_pipeline.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Level 3 Notebook created.")
