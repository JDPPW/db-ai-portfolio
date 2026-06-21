"""
FastAPI Backend — Weather Intelligence Agent
Exposes the MCP-powered agent via REST API
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import process_query

app = FastAPI(
    title="Weather Intelligence Agent",
    description="MCP-powered weather intelligence with real-time data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class QueryRequest(BaseModel):
    message: str

class QueryResponse(BaseModel):
    response: str
    tools_used: int
    tool_calls: list

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    base_dir = os.path.dirname(__file__)
    ui_path = os.path.join(base_dir, "ui.html")
    with open(ui_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    result = await process_query(request.message)
    return QueryResponse(**result)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "system": "Weather Intelligence Agent",
        "mcp_tools": ["get_current_weather", "get_forecast", "compare_locations"],
        "data_source": "OpenWeatherMap API (real-time)"
    }