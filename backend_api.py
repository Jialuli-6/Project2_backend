from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List

from backend import run_chat_turn

app = FastAPI(title="SciSciNet Chatbot")

# Frontend (Vite) call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Request body from the frontend."""
    query: str


class ChatResponse(BaseModel):
    """Shape of the response the frontend expects."""
    reply: str
    vega_specs: List[Dict[str, Any]]
    title: str
    plan: Dict[str, Any]
    grouped_preview: List[Dict[str, Any]]


@app.post("/api/sciscinet-chat", response_model=ChatResponse)
def sciscinet_chat(req: ChatRequest) -> ChatResponse:

    result = run_chat_turn(req.query)

    return ChatResponse(
        reply=result["reply"],
        # backend.py returns a single vega_spec; wrap it as a list for frontend
        vega_specs=[result["vega_spec"]],
        title=result["title"],
        plan=result["plan"],
        grouped_preview=result["grouped_preview"],
    )
