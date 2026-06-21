"""
FastAPI Backend — TechCore Enterprise IT Knowledge Agent
Agentic RAG with Self-Correction Loop
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import process_question

app = FastAPI(
    title="TechCore IT Knowledge Agent",
    description="Enterprise IT knowledge base with self-correcting RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    confidence_score: int
    correction_count: int
    grade: str
    grade_reason: str

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    base_dir = os.path.dirname(__file__)
    ui_path = os.path.join(base_dir, "ui.html")
    with open(ui_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    result = process_question(request.question)
    return AnswerResponse(**result)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "system": "TechCore IT Knowledge Agent",
        "self_correction": "enabled",
        "max_correction_loops": 3
    }