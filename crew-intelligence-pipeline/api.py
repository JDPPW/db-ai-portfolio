"""
FastAPI Backend — Enterprise Intelligence Pipeline
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pipeline import run_intelligence_pipeline

app = FastAPI(
    title="Enterprise Intelligence Pipeline",
    description="Multi-agent intelligence reports powered by CrewAI + LangGraph",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ReportRequest(BaseModel):
    topic: str

class ReportResponse(BaseModel):
    topic: str
    report: str
    status: str
    quality_grade: str
    quality_reason: str
    retry_count: int

@app.post("/generate", response_model=ReportResponse)
async def generate(request: ReportRequest):
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty")

    if len(request.topic) > 200:
        raise HTTPException(status_code=400, detail="Topic too long — keep it under 200 characters")

    import asyncio
    result = await asyncio.get_event_loop().run_in_executor(
        None, run_intelligence_pipeline, request.topic
    )
    return ReportResponse(**result)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "pipeline": "CrewAI + LangGraph",
        "agents": ["Research Specialist", "Strategic Analyst", "Report Writer"],
        "orchestration": "LangGraph state machine with quality gates"
    }