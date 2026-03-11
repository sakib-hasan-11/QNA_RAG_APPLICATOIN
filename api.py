"""
FastAPI Backend for RAG Query System
"""

import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipelines.pipeline import RAGPipeline

app = FastAPI(title="RAG Query API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline globally
pipeline = None


class QueryRequest(BaseModel):
    query: str
    k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: list
    query: str


@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup"""
    global pipeline
    try:
        pipeline = RAGPipeline()
        pipeline.setup_models()
        pipeline.setup_database()
        print("✓ RAG Pipeline initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing pipeline: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "online", "service": "RAG Query API"}


@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the RAG system

    Args:
        request: QueryRequest with query text and optional k parameter

    Returns:
        QueryResponse with answer and sources
    """
    try:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        result = pipeline.query_pipeline(
            query=request.query, k=request.k, verbose=False
        )

        return QueryResponse(
            answer=result["answer"].content
            if hasattr(result["answer"], "content")
            else str(result["answer"]),
            sources=result["sources"],
            query=request.query,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
