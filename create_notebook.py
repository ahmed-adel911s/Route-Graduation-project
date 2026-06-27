import nbformat as nbf

nb = nbf.v4.new_notebook()

text1 = """# Level 1: Foundation RAG Pipeline

This notebook implements a basic Retrieval-Augmented Generation (RAG) pipeline for the Azure documentation dataset.
We will load the JSON dataset, chunk it, create a vector database, and use the Google Gemini API to answer questions."""

code1 = """import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 1. Load the dataset
with open('dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} documents from dataset.json")"""

text2 = """## Chunking Strategy

We will use the **RecursiveCharacterTextSplitter** from LangChain.
Because this is technical documentation, we split it into fixed-size chunks of 1000 characters with a 150-character overlap to preserve context between chunks."""

code2 = """from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create Document objects
docs = []
for item in dataset:
    doc = Document(
        page_content=item['content'],
        metadata={"title": item['title'], "url": item['url']}
    )
    docs.append(doc)

# Initialize splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)

# Split documents
chunks = text_splitter.split_documents(docs)
print(f"Split {len(dataset)} documents into {len(chunks)} chunks.")
print(f"Sample chunk:\\n{chunks[0].page_content}")"""

text3 = """## Embedding & Indexing

We will convert our chunks into numerical vectors (embeddings) and store them in a FAISS vector database.
We'll use a local HuggingFace embedding model (`all-MiniLM-L6-v2`) which is free and fast."""

code3 = """from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS vector store
print("Building vector database... this may take a minute.")
vector_store = FAISS.from_documents(chunks, embeddings)

# Save it locally so we don't have to rebuild it every time
vector_store.save_local("faiss_index")
print("Vector database saved to 'faiss_index' directory.")"""

text4 = """## Basic Retrieval

Let's test if our vector database can find relevant chunks for a specific query."""

code4 = """# Load the vector store (optional, if running in a new session)
# vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

query = "What is Azure Chaos Studio?"
print(f"Query: {query}\\n")

# Retrieve top 3 most similar chunks
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
retrieved_docs = retriever.invoke(query)

for i, doc in enumerate(retrieved_docs):
    print(f"--- Chunk {i+1} ---")
    print(f"Metadata: {doc.metadata}")
    print(f"Content snippet: {doc.page_content[:200]}...\\n")"""

text5 = """## Answer Generation Using Gemini API

Now we pass the retrieved chunks and the user's question to the Google Gemini API to generate an answer."""

code5 = """from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# Create a prompt template instructing the model to ONLY use the provided context
template = \"\"\"
You are an expert AI assistant answering questions about Azure documentation.
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Do NOT hallucinate.
Try to keep the answer concise and well-formatted.

Context:
{context}

Question: {question}

Answer:
\"\"\"

prompt = PromptTemplate.from_template(template)

# Format the retrieved documents into a single string
def format_docs(docs):
    return "\\n\\n".join(doc.page_content for doc in docs)

# Create the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Test the full pipeline
question = "What is Azure Chaos Studio?"
print(f"Question: {question}\\n")

response = rag_chain.invoke(question)

print("=== FINAL ANSWER ===")
print(response)

print("\\n=== SOURCES ===")
for doc in retrieved_docs:
    print(f"- {doc.metadata['title']} ({doc.metadata['url']})")"""

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

with open('level1_rag.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created.")
