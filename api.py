"""REST API for the local RAG demo."""

import logging
import os
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_engine import RAGEngine

logger = logging.getLogger(__name__)
app = FastAPI(title="Internal Knowledge AI Assistant", version="1.1.0")


@lru_cache(maxsize=1)
def get_engine() -> RAGEngine:
    # Load once on the first question, not during API import or /health.
    return RAGEngine(data_dir=os.getenv("RAG_DATA_DIR") or None)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: int = Field(default=3, ge=1, le=5)


class SourceItem(BaseModel):
    source: str
    chunk_id: int
    page: int | None
    chunk: str
    similarity: float


class AskResponse(BaseModel):
    answer: str
    evidence: SourceItem | None
    sources: list[SourceItem]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, engine: RAGEngine = Depends(get_engine)) -> dict:
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be blank")
    try:
        return engine.answer(request.question, request.top_k)
    except Exception:
        logger.exception("Question answering failed")
        raise HTTPException(status_code=500, detail="Question answering failed") from None
