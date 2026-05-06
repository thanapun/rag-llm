"""
search_faiss.py — FAISS vector search worker
เรียกใช้โดย search.php ผ่าน shell_exec

Usage:
    python3 search_faiss.py '<json_payload>'

JSON payload:
    {
        "query":         "รหัสผ่านหมดอายุทำอย่างไร",
        "embedding_dir": "/var/www/html/embedding/",
        "top_k":         5,
        "model":         "BAAI/bge-m3"   // optional
    }

Output (stdout, last line):
    { "chunks": [ { "text": "...", "source": "...", "score": 0.92 }, ... ] }
"""

from __future__ import annotations

import json
import sys
import os
import pickle
import logging

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.WARNING)   # ไม่ให้ log รบกวน stdout

# ---- Cache model ใน process (ถ้า PHP ใช้ persistent worker จะเร็วขึ้น) ----
_model_cache: dict[str, SentenceTransformer] = {}


def get_model(name: str) -> SentenceTransformer:
    if name not in _model_cache:
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]


def search(
    query: str,
    embedding_dir: str,
    top_k: int = 5,
    model_name: str = "BAAI/bge-m3",
) -> list[dict]:
    """ค้นหา top-k chunks ที่เกี่ยวข้องกับ query"""

    # โหลด index + sidecar files
    index    = faiss.read_index(os.path.join(embedding_dir, "faiss.index"))
    with open(os.path.join(embedding_dir, "faiss_metadata.pkl"), "rb") as f:
        metadatas: list[dict] = pickle.load(f)
    with open(os.path.join(embedding_dir, "faiss_texts.pkl"), "rb") as f:
        texts: list[str] = pickle.load(f)

    # Embed query (prefix "query: " ตาม bge-m3 / E5 convention)
    model       = get_model(model_name)
    q_vec: np.ndarray = model.encode(
        [f"query: {query.strip()}"],
        normalize_embeddings=True,
    )

    # ค้นหา
    scores, indices = index.search(q_vec, top_k)

    chunks = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:          # FAISS คืน -1 เมื่อ index ว่าง
            continue
        raw_text = texts[idx]
        # ตัด prefix "passage: " ออกก่อนส่งกลับ
        clean_text = raw_text.removeprefix("passage: ").strip()
        chunks.append({
            "text":   clean_text,
            "source": metadatas[idx].get("source", "unknown"),
            "sheet":  metadatas[idx].get("sheet", ""),    # สำหรับ Excel
            "score":  round(float(score), 4),
        })

    return chunks


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No payload argument"}))
        sys.exit(1)

    try:
        payload = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON payload: {e}"}))
        sys.exit(1)

    query         = payload.get("query", "").strip()
    embedding_dir = payload.get("embedding_dir", "./embedding/")
    top_k         = int(payload.get("top_k", 5))
    model_name    = payload.get("model", "BAAI/bge-m3")

    if not query:
        print(json.dumps({"error": "query is empty"}))
        sys.exit(1)

    try:
        chunks = search(query, embedding_dir, top_k, model_name)
        # พิมพ์ผลลัพธ์เป็น JSON บรรทัดสุดท้าย (PHP อ่าน end($lines))
        print(json.dumps({"chunks": chunks}, ensure_ascii=False))
    except FileNotFoundError as e:
        print(json.dumps({"error": f"Index file not found: {e}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
