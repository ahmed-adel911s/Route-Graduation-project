import nbformat as nbf

nb = nbf.v4.new_notebook()

text1 = """# Level 2: Advanced RAG (Query Expansion)

This notebook upgrades our RAG pipeline by implementing **Multi-Query (Query Expansion)** from scratch.
Instead of relying on a single user query, we use the LLM to explicitly generate 3 different variations of the question. 
We then retrieve documents for all variations, combine them, remove duplicates, and generate a final comprehensive answer. Building this logic from scratch makes it robust and transparent for the graduation project!"""

code1 = """import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Setup & API Keys
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize models
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)"""

text2 = """## Load Existing Vector Database

We don't need to rebuild our database. We can load the `faiss_index` we created in Level 1!"""

code2 = """# Load FAISS database from disk
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Base retriever
base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})"""

text3 = """## Create the Custom Multi-Query Retriever

We will build a custom pipeline that uses Gemini to rewrite the user's question into 3 variations, searches the database for all of them, and merges the unique results!"""

code3 = """# 1. Prompt to generate multiple queries
multi_query_template = \"\"\"You are an AI language model assistant. Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search. Provide these alternative questions separated by newlines. 
Original question: {question}\"\"\"

prompt_perspectives = PromptTemplate.from_template(multi_query_template)

# 2. Chain to generate the queries
generate_queries = (
    prompt_perspectives 
    | llm 
    | StrOutputParser() 
    | (lambda x: x.split('\\n'))
)

# 3. Custom retrieval function
def get_unique_docs(query):
    queries = generate_queries.invoke({"question": query})
    # Filter out empty strings and clean up the list
    queries = [q.strip() for q in queries if q.strip()]
    
    print("=== EXPANDED QUERIES ===")
    for i, q in enumerate(queries):
        print(f"{i+1}. {q}")
    print("========================\\n")
    
    unique_docs = {}
    # Search for all generated queries PLUS the original query
    for q in queries + [query]:
        docs = base_retriever.invoke(q)
        for doc in docs:
            # Use page_content as a simple unique key
            unique_docs[doc.page_content] = doc
            
    return list(unique_docs.values())

query = "What are Azure Dedicated Hosts and how do they work?"
print(f"Original Query: {query}\\n")

# Let's test the retrieval step
retrieved_docs = get_unique_docs(query)

print(f"Retrieved {len(retrieved_docs)} unique chunks across all query variations!\\n")
for i, doc in enumerate(retrieved_docs):
    print(f"--- Unique Chunk {i+1} ---")
    print(f"Title: {doc.metadata['title']}")
    print(f"URL: {doc.metadata['url']}")
    print(f"Snippet: {doc.page_content[:150]}...\\n")"""

text4 = """## Answer Generation

Now we pass all the retrieved unique chunks to Gemini to answer the original question."""

code4 = """# Create Answer Prompt Template
template = \"\"\"
You are an expert AI assistant answering questions about Azure documentation.
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Do NOT hallucinate.
Provide a detailed and well-formatted answer.

Context:
{context}

Question: {question}

Answer:
\"\"\"

prompt = PromptTemplate.from_template(template)

def retrieve_and_format(query):
    docs = get_unique_docs(query)
    return "\\n\\n".join(doc.page_content for doc in docs)

# Create the advanced RAG chain
advanced_rag_chain = (
    {"context": retrieve_and_format, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("=== FINAL GENERATED ANSWER ===\\n")
# Test a completely new query for the final pipeline!
final_query = "What is Azure Data Explorer used for?"
print(f"Testing Pipeline with: {final_query}\\n")
response = advanced_rag_chain.invoke(final_query)
print(response)"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text1),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_markdown_cell(text2),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(text3),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_markdown_cell(text4),
    nbf.v4.new_code_cell(code4)
]

with open('level2_rag_advanced.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Level 2 Custom Notebook created.")
