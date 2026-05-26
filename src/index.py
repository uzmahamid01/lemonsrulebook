"""Build/refresh the Chroma vector index from chunks.json.

Embeddings come from Voyage AI's `voyage-4-large` (1024d, top of MTEB IR for
question->passage matching). Documents are embedded with input_type="document";
queries (in qa.py) use input_type="query" for asymmetric retrieval.

Chroma metadata values must be primitives; image_paths is a list so we
pipe-join it for storage and split it back at query time (see qa.py).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import chromadb
import voyageai
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "chunks.json"
IMAGES_FILE = ROOT / "images.json"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "lemons"
IMAGES_COLLECTION = "lemons_images"
FORUM_COLLECTION_NAME = "lemons_forum"
MODEL_NAME = "voyage-4-large"            # text-only chunks
IMAGE_MODEL_NAME = "voyage-multimodal-3"  # embeds actual image content (+ optional text companion)


EMBED_BATCH = 30
SLEEP_BETWEEN_BATCHES_SEC = 22
IMAGE_EMBED_BATCH = 2
IMAGE_SLEEP_SEC = 25

_voyage: voyageai.Client | None = None

def voyage() -> voyageai.Client:
    """Lazy singleton so importing this module doesn't require VOYAGE_API_KEY."""
    global _voyage
    if _voyage is None:
        _voyage = voyageai.Client()
    return _voyage


def _serialize_meta(c: dict) -> dict:
    """Convert chunk metadata to Chroma-compatible primitives (join lists with pipes)."""
    return {
        "doc": c["doc"],
        "doc_file": c["doc_file"],
        "citation": c["citation"],
        "rule_number": c["rule_number"] or "",
        "title": c["title"],
        "page": c["page"],
        "image_paths": "|".join(c["image_paths"]),
    }


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunks with input_type='document'. Throttles between batches
    to stay under voyage free-tier rate limits (3 RPM / 10K TPM)."""
    out: list[list[float]] = []
    n_batches = (len(texts) + EMBED_BATCH - 1) // EMBED_BATCH
    for bi, i in enumerate(range(0, len(texts), EMBED_BATCH)):
        batch = texts[i : i + EMBED_BATCH]
        if bi > 0:
            print(f"  ...waiting {SLEEP_BETWEEN_BATCHES_SEC}s for rate limit")
            time.sleep(SLEEP_BETWEEN_BATCHES_SEC)
        print(f"  batch {bi + 1}/{n_batches}: embedding {len(batch)} chunks")
        # Simple retry on rate-limit error.
        for attempt in range(3):
            try:
                res = voyage().embed(
                    batch, model=MODEL_NAME, input_type="document", truncation=True
                )
                out.extend(res.embeddings)
                break
            except voyageai.error.RateLimitError:
                wait = 30 * (attempt + 1)
                print(f"    rate-limited, sleeping {wait}s then retrying")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed after retries on batch {bi + 1}")
    return out


def embed_query(text: str) -> list[float]:
    """Embed a single user question with input_type='query'."""
    res = voyage().embed(
        [text], model=MODEL_NAME, input_type="query", truncation=True
    )
    return res.embeddings[0]


def _open_image(path: Path):
    """Lazy PIL import so the module loads without Pillow for non-image flows."""
    from PIL import Image
    return Image.open(path).convert("RGB")


def embed_multimodal_documents(items: list[list]) -> list[list[float]]:
    """Embed multimodal inputs with voyage-multimodal-3, input_type='document'.

    Each item is a list mixing text strings and PIL Image objects. For our
    corpus we pair each image with its page text so the embedding fuses
    visual content with textual context. Throttled with a smaller batch and
    longer sleep than the text-only path (voyage-multimodal-3 has tighter
    per-call pixel limits on the free tier).
    """
    out: list[list[float]] = []
    n_batches = (len(items) + IMAGE_EMBED_BATCH - 1) // IMAGE_EMBED_BATCH
    for bi, i in enumerate(range(0, len(items), IMAGE_EMBED_BATCH)):
        batch = items[i : i + IMAGE_EMBED_BATCH]
        if bi > 0:
            print(f"  ...waiting {IMAGE_SLEEP_SEC}s for rate limit")
            time.sleep(IMAGE_SLEEP_SEC)
        print(f"  batch {bi + 1}/{n_batches}: embedding {len(batch)} multimodal items")
        for attempt in range(3):
            try:
                res = voyage().multimodal_embed(
                    inputs=batch, model=IMAGE_MODEL_NAME, input_type="document",
                )
                out.extend(res.embeddings)
                break
            except voyageai.error.RateLimitError:
                wait = 30 * (attempt + 1)
                print(f"    rate-limited, sleeping {wait}s then retrying")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed after retries on multimodal batch {bi + 1}")
    return out


def embed_multimodal_query(text: str | None = None, image=None) -> list[float]:
    """Embed a query that may include text, an image, or both.
    Returns one embedding from voyage-multimodal-3."""
    parts: list = []
    if text and text.strip():
        parts.append(text.strip())
    if image is not None:
        parts.append(image)
    if not parts:
        raise ValueError("embed_multimodal_query needs text or image (or both)")
    res = voyage().multimodal_embed(
        inputs=[parts], model=IMAGE_MODEL_NAME, input_type="query",
    )
    return res.embeddings[0]


def build_index(reset: bool = True):
    """Load chunks, embed them with Voyage AI, and populate the Chroma text-chunks collection."""
    chunks = json.loads(CHUNKS_FILE.read_text())
    print(f"[index] loaded {len(chunks)} chunks from {CHUNKS_FILE.name}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    coll = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metas = [_serialize_meta(c) for c in chunks]

    print(f"[index] embedding {len(texts)} chunks with {MODEL_NAME} (document)...")
    embeddings = embed_documents(texts)
    print(f"[index] got {len(embeddings)} vectors, dim={len(embeddings[0])}")

    coll.add(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)
    print(f"[index] persisted -> {CHROMA_DIR.relative_to(ROOT)}")
    print(f"[index] collection size: {coll.count()}")

    # Smoke test: the two example questions from the brief.
    for q in [
        "Our team would like to put in a stronger transmission. Would this be allowed?",
        "Can I use the sale of the old transmission to offset that price?",
    ]:
        qv = [embed_query(q)]
        res = coll.query(query_embeddings=qv, n_results=3)
        print(f"\nQ: {q}")
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            print(f"  - {meta['citation']:24s}  dist={dist:.3f}  | {meta['title'][:60]}")

    return coll


def build_image_index(reset: bool = True):
    """Embed images (with their page-text companion) into a Chroma collection.

    Uses voyage-multimodal-3 to embed the actual image content fused with
    its page text, instead of embedding only the page text. That means
    retrieval can match by visual content (and a user-uploaded photo can
    match against this same vector space).
    """
    if not IMAGES_FILE.exists():
        print(f"[image-index] {IMAGES_FILE.name} not found; run src.ingest first")
        return None
    raw = json.loads(IMAGES_FILE.read_text())
    images = [m for m in raw if (ROOT / m["path"]).exists()]
    print(f"[image-index] {len(images)} images to embed")
    if not images:
        return None

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(IMAGES_COLLECTION)
        except Exception:
            pass
    coll = client.get_or_create_collection(
        name=IMAGES_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [m["path"] for m in images]
    metas = [
        {
            "path": m["path"],
            "doc": m["doc"],
            "page": m["page"],
            "caption": m["caption"],
            "doc_file": m.get("doc_file", ""),  # filename for context
        }
        for m in images
    ]
    docs = [m["caption"] or m["path"] for m in images]  # what Chroma stores as the "document"

    # Build multimodal inputs: [caption_text, PIL_Image] when caption exists,
    # [PIL_Image] otherwise. Each input produces one fused embedding.
    inputs = []
    for m in images:
        pil = _open_image(ROOT / m["path"])
        if m["caption"]:
            inputs.append([m["caption"], pil])
        else:
            inputs.append([pil])

    print(f"[image-index] embedding {len(inputs)} images with {IMAGE_MODEL_NAME}...")
    embeddings = embed_multimodal_documents(inputs)
    coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    print(f"[image-index] persisted, size: {coll.count()}")
    return coll


if __name__ == "__main__":
    build_index()
    build_image_index()
