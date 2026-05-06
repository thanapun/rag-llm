"""
search_api.py — FAISS Vector Search · FastAPI Service

Run:
    pip install fastapi uvicorn sentence-transformers faiss-cpu numpy
    python search_api.py

Or with uvicorn directly:
    uvicorn search_api:app --host 0.0.0.0 --port 8000 --workers 1

Endpoints:
    GET  /health          — liveness check
    POST /search          — vector search
    GET  /docs            — Swagger UI (auto-generated)
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# โหลด .env อัตโนมัติ (ถ้ามี) — ไม่ error ถ้าไม่มีไฟล์
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv ไม่ได้ติดตั้ง — ใช้ env variable ปกติแทน

import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config (environment variables → defaults)
# ---------------------------------------------------------------------------

EMBEDDING_DIR = Path(os.getenv("EMBEDDING_DIR", "./embedding"))
MODEL_NAME    = os.getenv("EMBED_MODEL",    "BAAI/bge-m3")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
HOST          = os.getenv("HOST", "0.0.0.0")
PORT          = int(os.getenv("PORT", "4500"))
LOG_LEVEL     = os.getenv("LOG_LEVEL", "info")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("search_api")

# ---------------------------------------------------------------------------
# In-memory index store (loaded once at startup)
# ---------------------------------------------------------------------------

class IndexStore:
    index:     faiss.Index | None = None
    metadatas: list[dict[str, Any]] = []
    texts:     list[str] = []
    model:     SentenceTransformer | None = None
    ready:     bool = False
    loaded_at: float = 0.0


store = IndexStore()


def load_index(embedding_dir: Path, model_name: str) -> None:
    """Load FAISS index + sidecar files + embedding model into memory."""
    logger.info("Loading FAISS index from: %s", embedding_dir)

    index_path = embedding_dir / "faiss.index"
    meta_path  = embedding_dir / "faiss_metadata.pkl"
    texts_path = embedding_dir / "faiss_texts.pkl"

    for p in (index_path, meta_path, texts_path):
        if not p.exists():
            raise FileNotFoundError(f"Required index file not found: {p}")

    store.index = faiss.read_index(str(index_path))
    with open(meta_path,  "rb") as f:
        store.metadatas = pickle.load(f)
    with open(texts_path, "rb") as f:
        store.texts = pickle.load(f)

    logger.info("Loading embedding model: %s", model_name)
    store.model = SentenceTransformer(model_name)

    store.ready     = True
    store.loaded_at = time.time()
    logger.info(
        "Index ready — %d vectors, dim=%d",
        store.index.ntotal,
        store.index.d,
    )


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_index(EMBEDDING_DIR, MODEL_NAME)
    except FileNotFoundError as e:
        logger.error("Startup failed: %s", e)
        logger.error("Run build_index.py first, then restart this service.")
        # ยัง start ได้ แต่ /health จะคืน 503
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TVO FAISS Search API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # จำกัด origin ใน production จริง
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1500, description="คำค้นหา")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20, description="จำนวน chunks ที่ต้องการ")


class ChunkResult(BaseModel):
    text:   str
    source: str
    sheet:  str
    score:  float


class SearchResponse(BaseModel):
    query:    str
    chunks:   list[ChunkResult]
    elapsed_ms: float


class HealthResponse(BaseModel):
    status:     str
    ready:      bool
    vectors:    int
    model:      str
    loaded_at:  float | None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, summary="Liveness & readiness check")
def health():
    if not store.ready:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded — run build_index.py first.",
        )
    return HealthResponse(
        status    = "ok",
        ready     = store.ready,
        vectors   = store.index.ntotal if store.index else 0,
        model     = MODEL_NAME,
        loaded_at = store.loaded_at,
    )


@app.post("/search", response_model=SearchResponse, summary="Vector similarity search")
def search(req: SearchRequest):
    if not store.ready:
        raise HTTPException(status_code=503, detail="Index not ready.")

    t0 = time.perf_counter()

    # Embed query (prefix ตาม bge-m3 / E5 convention)
    q_vec: np.ndarray = store.model.encode(
        [f"query: {req.query.strip()}"],
        normalize_embeddings=True,
    )

    # ค้นหา
    scores, indices = store.index.search(q_vec, req.top_k)

    chunks: list[ChunkResult] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        raw_text = store.texts[idx]
        meta     = store.metadatas[idx]
        chunks.append(ChunkResult(
            text   = raw_text.removeprefix("passage: ").strip(),
            source = meta.get("source", "unknown"),
            sheet  = meta.get("sheet",  ""),
            score  = round(float(score), 4),
        ))

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("query=%r  top_k=%d  results=%d  %.1fms", req.query, req.top_k, len(chunks), elapsed_ms)

    return SearchResponse(query=req.query, chunks=chunks, elapsed_ms=elapsed_ms)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "search_api:app",
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL,
        reload=False,   # ปิด reload ใน production
    )
