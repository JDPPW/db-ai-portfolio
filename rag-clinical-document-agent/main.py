"""
RAG Clinical Document Intelligence Agent
Author: Dujuan Brown | Iron Gate Solutions US LLC
GitHub: github.com/JDPPW/db-ai-portfolio

Ingests a clinical document, embeds it into ChromaDB,
and answers questions with cited source context.

Demonstrates: RAG pipeline, vector search, grounded generation
Target roles: MedOne Systems, VetsEZ, Katalyst Healthcare
"""

import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Load API key from .env ────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("ERROR: ANTHROPIC_API_KEY not found.")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
DOCUMENT_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "patient_intake.txt")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
MODEL = "claude-haiku-4-5-20251001"

# ── Step 1: Load and chunk the document ──────────────────────────────────────
def load_and_chunk(path):
    print(f"\n📄 Loading document: {path}")
    loader = TextLoader(path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    print(f"   Split into {len(chunks)} chunks")
    return chunks

# ── Step 2: Embed and store in ChromaDB ──────────────────────────────────────
def build_vectorstore(chunks):
    print("\n🔢 Building vector store...")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=None,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"   Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore

# ── Step 3: Build RAG chain ───────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    llm = ChatAnthropic(
        model=MODEL,
        api_key=api_key,
        temperature=0
    )
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )
    
    prompt = PromptTemplate.from_template("""
You are a clinical AI assistant helping healthcare staff quickly retrieve 
information from patient documents. Answer using ONLY the context below.
If the answer is not in the context, say "That information is not in this document."
Always be specific. This is a HIPAA-controlled environment.

Context:
{context}

Question: {input}

Answer:""")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# ── Step 4: Run Q&A session ───────────────────────────────────────────────────
def run_session(rag_chain):
    print("\n" + "="*60)
    print("  Clinical Document Intelligence Agent")
    print("  HIPAA Notice: Authorized access only")
    print("  Type 'quit' to exit")
    print("="*60)
    
    while True:
        question = input("\n❓ Clinical query: ").strip()
        
        if question.lower() in ["quit", "exit", "q"]:
            print("\nSession closed.")
            break
            
        if not question:
            continue
        
        print("\n🔍 Retrieving relevant context...\n")
        result = rag_chain.invoke(question)
        print(f"💬 Answer:\n{result}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    chunks = load_and_chunk(DOCUMENT_PATH)
    vectorstore = build_vectorstore(chunks)
    rag_chain = build_rag_chain(vectorstore)
    run_session(rag_chain)

if __name__ == "__main__":
    main()