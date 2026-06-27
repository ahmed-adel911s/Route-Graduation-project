import nbformat as nbf

nb = nbf.v4.new_notebook()

text1 = """# Level 2: Advanced RAG (Query Expansion)

This notebook upgrades our RAG pipeline by implementing **Multi-Query (Query Expansion)**.
Instead of relying on a single user query, we use the LLM to generate 3 different variations of the question. 
We then retrieve documents for all 3 variations, combine them, and generate a final comprehensive answer."""

code1 = """import os
import logging
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Setup & API Keys
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set up logging to see the generated queries!
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# Initialize models
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)"""

text2 = """## Load Existing Vector Database

We don't need to rebuild our database. We can load the `faiss_index` we created in Level 1!"""

code2 = """# Load FAISS database from disk
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Base retriever
base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})"""

text3 = """## Create the Multi-Query Retriever

We wrap our base retriever in a `MultiQueryRetriever`. It uses Gemini to generate multiple versions of the question."""

code3 = """# Initialize the advanced retriever
advanced_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever, 
    llm=llm
)

query = "What are Azure Dedicated Hosts and how do they work?"
print(f"Original Query: {query}\\n")

# Let's test the retrieval step (check the logs for the 3 generated variations!)
retrieved_docs = advanced_retriever.invoke(query)

print(f"\\nRetrieved {len(retrieved_docs)} unique chunks across all query variations!\\n")
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

def format_docs(docs):
    return "\\n\\n".join(doc.page_content for doc in docs)

# Create the advanced RAG chain
advanced_rag_chain = (
    {"context": advanced_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("=== FINAL GENERATED ANSWER ===\\n")
response = advanced_rag_chain.invoke(query)
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
print("Level 2 Notebook created.")
