"""Embed forum_chunks.json into the lemons_forum Chroma collection.

Sibling of src.index.build_index. The vector is computed from the simplified
Q&A text (each chunk's `text` field is already "Q: ...\\nA: ..."); the
original posts are preserved in metadata so the LLM can ground its
plain-English framing in real community discussion.

Run:
    uv run python -m src.index_forum
"""
from __future__ import annotations

import json
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from src.index import (
    CHROMA_DIR,
    FORUM_COLLECTION_NAME,
    MODEL_NAME,
    embed_documents,
    embed_query,
)

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
FORUM_CHUNKS_FILE = ROOT / "forum_chunks.json"


def _serialize_meta_forum(c: dict) -> dict:
    """Forum chunks have no image_paths and no rule_number; everything is primitives."""
    return {
        "doc": c["doc"],
        "doc_file": c["doc_file"],
        "citation": c["citation"],
        "title": c["title"],
        "thread_url": c.get("thread_url", ""),
        "thread_id": c.get("thread_id", ""),
        "author": c.get("author", ""),
        "creation_time": c.get("creation_time", ""),
        "source_file": c.get("source_file", ""),
        "original_post_count": c.get("original_post_count", 0),
        "original_text_truncated": c.get("original_text_truncated", ""),
    }


def build_forum_index(reset: bool = True):
    """Load forum_chunks.json, embed each Q&A with Voyage, and populate the lemons_forum collection."""
    if not FORUM_CHUNKS_FILE.exists():
        print(f"[forum-index] {FORUM_CHUNKS_FILE.name} not found; run src.ingest_forum first")
        return None
    chunks = json.loads(FORUM_CHUNKS_FILE.read_text())
    print(f"[forum-index] loaded {len(chunks)} forum chunks from {FORUM_CHUNKS_FILE.name}")
    if not chunks:
        return None

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(FORUM_COLLECTION_NAME)
        except Exception:
            pass
    coll = client.get_or_create_collection(
        name=FORUM_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metas = [_serialize_meta_forum(c) for c in chunks]

    print(f"[forum-index] embedding {len(texts)} Q&A docs with {MODEL_NAME} (document)...")
    embeddings = embed_documents(texts)
    print(f"[forum-index] got {len(embeddings)} vectors, dim={len(embeddings[0])}")

    coll.add(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)
    print(f"[forum-index] persisted -> {CHROMA_DIR.relative_to(ROOT)}")
    print(f"[forum-index] collection size: {coll.count()}")

    # Smoke probe: one plain-English question, see what comes back.
    q = "What roll cage tubing diameter do most teams actually use?"
    qv = [embed_query(q)]
    res = coll.query(query_embeddings=qv, n_results=3)
    print(f"\nQ: {q}")
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        print(f"  - {meta['citation'][:60]:60s}  dist={dist:.3f}")

    return coll


if __name__ == "__main__":
    build_forum_index()
