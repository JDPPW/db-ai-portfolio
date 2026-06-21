"""
FastAPI Backend — RetailCore Customer Service Agent
Exposes REST endpoints consumed by the browser UI.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import process_message, BRANDS

app = FastAPI(
    title="RetailCore Customer Service Agent API",
    description="Multi-tenant AI customer service platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Request/Response models ───────────────────────────────────────────────────
class MessageRequest(BaseModel):
    brand_id: str
    message: str
    conversation_history: list = []
    frustration_count: int = 0
    exchange_count: int = 0

class MessageResponse(BaseModel):
    brand_name: str
    decision: str
    response: str
    case_summary: str = ""
    exchange_count: int = 0

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the chat interface."""
    base_dir = os.path.dirname(__file__)
    ui_path = os.path.join(base_dir, "ui.html")
    with open(ui_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/brands")
async def get_brands():
    """Return available brands for the UI dropdown."""
    return {
        brand_id: {
            "name": info["name"],
            "color": info["color"],
            "tagline": info["tagline"]
        }
        for brand_id, info in BRANDS.items()
    }

@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    """Process a customer message and return agent response."""
    if request.brand_id not in BRANDS:
        raise HTTPException(status_code=400, detail=f"Unknown brand: {request.brand_id}")
    
    result = process_message(
        brand_id=request.brand_id,
        customer_message=request.message,
        conversation_history=request.conversation_history,
        frustration_count=request.frustration_count,
        exchange_count=request.exchange_count
    )
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return MessageResponse(**result)

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "platform": "RetailCore Customer Service Agent"}