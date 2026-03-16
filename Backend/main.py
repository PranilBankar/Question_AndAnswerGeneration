"""
main.py — FastAPI application entry point for the NEET Biology Q&A system

Run:
    uvicorn Backend.main:app --reload --port 8000

Docs:
    http://localhost:8000/docs    (Swagger UI)
    http://localhost:8000/redoc   (ReDoc)
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Backend.schemas import HealthResponse
from Backend.routers.qa import router as qa_router


# ==============================
# STARTUP / SHUTDOWN
# ==============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-load heavy models at startup so the first request is fast.
    """
    print("\n🚀 Starting NEET Biology Q&A Backend...")
    start = time.time()

    # Pre-load BERT classifier (singleton)
    from Classification.classify_and_extract import get_predictor
    get_predictor()

    # Pre-load embedding model (singleton)
    from rag.retriever import get_embedding_model
    get_embedding_model()

    elapsed = time.time() - start
    print(f"✅ All models loaded in {elapsed:.1f}s — server ready!\n")

    yield  # App runs here

    print("\n🛑 Shutting down NEET Biology Q&A Backend.")


# ==============================
# APP INSTANCE
# ==============================
app = FastAPI(
    title="NEET Biology Q&A API",
    description=(
        "AI-powered Question Answering system for NEET Biology. "
        "Uses BERT classification for chapter detection and a "
        "RAG pipeline with NCERT textbook context for accurate answers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ==============================
# CORS — Allow frontend access
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",     # React/Next.js dev
        "http://localhost:5173",     # Vite dev
        "http://localhost:5500",     # Live Server
        "http://127.0.0.1:5500",
        "*",                         # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# ROUTES
# ==============================
app.include_router(qa_router)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
async def health_check():
    """Check if the server and all components are running."""
    from Classification.classify_and_extract import _predictor
    from rag.retriever import _embedding_model, _supabase_client

    return HealthResponse(
        status="ok",
        version="1.0.0",
        components={
            "bert_classifier": "loaded" if _predictor is not None else "not loaded",
            "embedding_model": "loaded" if _embedding_model is not None else "not loaded",
            "supabase": "connected" if _supabase_client is not None else "not connected",
        },
    )


@app.get("/", tags=["System"])
async def root():
    """API root — welcome message."""
    return {
        "message": "NEET Biology Q&A API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "POST /api/qa": "Full Q&A pipeline (classify + answer + verify)",
            "POST /api/classify": "Chapter classification + topic extraction only",
            "GET /health": "Health check",
        },
    }
