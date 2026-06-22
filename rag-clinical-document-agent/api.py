"""
FastAPI Backend — RAG Clinical Document Intelligence Agent
Author: Dujuan Brown | Iron Gate Solutions US LLC
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="RAG Clinical Document Intelligence Agent",
    description="HIPAA-aware clinical document Q&A powered by RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Build RAG chain on startup
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

api_key = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-haiku-4-5-20251001"

def build_rag_chain():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(base_dir, "sample_data", "patient_intake.txt")

    loader = TextLoader(doc_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=None,
        collection_name="clinical_docs",
        persist_directory=os.path.join(base_dir, "chroma_db")
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatAnthropic(model=MODEL, api_key=api_key, temperature=0)

    prompt = PromptTemplate.from_template("""You are a clinical AI assistant helping 
healthcare staff retrieve information from patient documents.
Answer using ONLY the context provided. If the answer is not in the context, 
say "That information is not in this document."
This is a HIPAA-controlled environment. Be specific and cite details.

Context:
{context}

Question: {input}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

rag_chain = build_rag_chain()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")
    with open(ui_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer = rag_chain.invoke(request.question)
    return QueryResponse(question=request.question, answer=answer)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "system": "RAG Clinical Document Intelligence Agent",
        "hipaa_aware": True
    }