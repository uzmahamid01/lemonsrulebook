"""Retrieve -> generate (Claude) -> validate citations.

The validator extracts every `Rule X.Y(.Z)` citation from the model's answer
and checks it against the known rule-number set from ingestion. A "valid"
answer cites only rules that actually exist in the corpus.

Embeddings: Voyage AI `voyage-3-large` (1024d). Strong enough on
question->passage matching to drop the previous HyDE expansion / multi-query
scaffold. Single retrieve call per question.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv

from src.index import (
    CHROMA_DIR,
    COLLECTION_NAME,
    IMAGES_COLLECTION,
    MODEL_NAME,
    embed_query,
    embed_multimodal_query,
)

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

ANTHROPIC_MODEL = "claude-sonnet-4-6"
TOP_K = 6
MAX_TOKENS = 700
IMAGE_TOP_K = 4
IMAGE_SIM_THRESHOLD = 0.55  # cosine distance; lower = more similar

SYSTEM_PROMPT = """\
You are the Lemons Virtual Inspector, helping racing teams understand the
24 Hours of Lemons rules. You will be given a question and a set of relevant
rules and images retrieved from the official documents.

RULES FOR YOUR ANSWER:
1. Answer using ONLY the information in the provided rules. Do not rely on
   outside knowledge or make up facts.
2. Every factual claim MUST be followed by an inline citation in one of these
   exact forms:
     (Rule X.Y) or (Rule X.Y.Z)        for rulebook rules
     (Safety Checklist p.N)             for the safety checklist
     (How Not To Fail Tech p.N)         for the how-not-to-fail guide
     (Tech Sheet)                       for the tech sheet
3. Do NOT invent rule numbers. Only cite rules that appear in the context.
4. If the provided rules and images don't cover the question, say so plainly and suggest
   the team contact an official Lemons inspector. Do not guess.
5. Keep answers concise: 2-5 sentences is typical. Use a bulleted list only
   when enumerating multiple distinct rules.
6. If the user attaches a photo, examine it and weave concrete observations
   into your answer ("the cage in your photo shows..."). Combine those visual
   observations with rule citations exactly as you would for a text-only
   question. If the photo seems unrelated to Lemons rules, say so plainly.
6. When referencing images, include a brief description of what the image shows
   and where it is located (e.g., "Image from Safety Checklist p.3").
7. If the questions asks for an image or diagram, include a description of what
   the image should show and where it is located (e.g., "Image from Safety Checklist p.3").
"""

# Captures parenthesized OR bare "Rule X.Y(.Z)" citations.
RULE_CITATION_RE = re.compile(r"Rule\s+(\d+(?:\.\d+){1,3})")


@dataclass
class Retrieved:
    citation: str
    text: str
    rule_number: str | None
    image_paths: list[str]
    page: int
    doc: str
    distance: float


@dataclass
class Answer:
    text: str
    sources: list[Retrieved]
    cited_rules: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    retrieved_images: list[dict] = field(default_factory=list)


# ---- module-level caches so Streamlit reruns don't reload everything ----
_collection: "chromadb.Collection | None" = None
_image_collection: "chromadb.Collection | None" = None
_valid_rule_numbers: set[str] = set()


def _resources():
    """Lazy-load the text chunks collection and the set of valid rule numbers."""
    global _collection, _valid_rule_numbers
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
        all_meta = _collection.get(include=["metadatas"])
        _valid_rule_numbers = {
            m["rule_number"] for m in all_meta["metadatas"] if m["rule_number"]
        }
    return _collection, _valid_rule_numbers


def _image_resources():
    """Lazy-load the image-caption collection; returns None if not built."""
    global _image_collection
    if _image_collection is None:
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            _image_collection = client.get_collection(IMAGES_COLLECTION)
        except Exception:
            _image_collection = None
    return _image_collection


def retrieve_images(
    query_text: str,
    image_bytes: bytes | None = None,
    k: int = IMAGE_TOP_K,
) -> list[dict]:
    """Find images that match the user's text question and/or uploaded photo.

    The image collection is embedded with voyage-multimodal-3, so queries
    can mix text and an image in the same call. When the user uploads a
    photo, we embed `[question_text, photo_pil]` together and look for
    similar diagrams from the rulebook. Text-only queries embed just text.
    """
    coll = _image_resources()
    if coll is None:
        return []
    pil = None
    if image_bytes:
        import io
        from PIL import Image
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if not query_text.strip() and pil is None:
        return []
    qv = [embed_multimodal_query(text=query_text or None, image=pil)]
    res = coll.query(query_embeddings=qv, n_results=k)
    out: list[dict] = []
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        if dist > IMAGE_SIM_THRESHOLD:
            continue
        out.append({**meta, "distance": float(dist)})
    return out


def retrieve(question: str, k: int = TOP_K) -> list[Retrieved]:
    """Single-stage dense retrieval with voyage-3-large query embedding."""
    coll, _ = _resources()
    qv = [embed_query(question)]
    res = coll.query(query_embeddings=qv, n_results=k)
    out: list[Retrieved] = []
    for meta, doc, dist in zip(
        res["metadatas"][0], res["documents"][0], res["distances"][0]
    ):
        out.append(
            Retrieved(
                citation=meta["citation"],
                text=doc,
                rule_number=meta["rule_number"] or None,
                image_paths=[p for p in meta["image_paths"].split("|") if p],
                page=meta["page"],
                doc=meta["doc"],
                distance=float(dist),
            )
        )
    return out


def _build_user_prompt(question: str, retrieved: list[Retrieved]) -> str:
    """Format the user's question and retrieved rules into a prompt for Claude."""
    parts = ["Relevant rules (cite these by their bracketed tag):\n"]
    for r in retrieved:
        parts.append(f"[{r.citation}]\n{r.text}\n")
    parts.append(f"\nQuestion: {question}")
    return "\n".join(parts)


def _validate(answer_text: str, valid: set[str]) -> tuple[list[str], list[str]]:
    """Extract cited rules from the answer and flag any that don't exist in the corpus."""
    cited = list(dict.fromkeys(RULE_CITATION_RE.findall(answer_text)))  # dedup, keep order
    invalid = [c for c in cited if c not in valid]
    return cited, invalid


def _build_user_content(user_prompt: str, image_bytes: bytes | None, image_mime: str):
    """Wrap the prompt as a text content block, optionally with an image block."""
    if not image_bytes:
        return user_prompt
    import base64
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    return [
        {"type": "text", "text": user_prompt},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": image_mime, "data": b64},
        },
    ]


def ask(
    question: str,
    image_bytes: bytes | None = None,
    image_mime: str = "image/jpeg",
    k: int = TOP_K,
) -> Answer:
    """Retrieve rules, generate an answer with Claude, validate citations, and surface related images."""
    _, valid = _resources()
    retrieved = retrieve(question, k=k)
    user_prompt = _build_user_prompt(question, retrieved)
    content = _build_user_content(user_prompt, image_bytes, image_mime)

    client = Anthropic()
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    text = "".join(block.text for block in msg.content if block.type == "text")
    cited, invalid = _validate(text, valid)
    images = retrieve_images(question, image_bytes=image_bytes)
    return Answer(
        text=text,
        sources=retrieved,
        cited_rules=cited,
        invalid_citations=invalid,
        retrieved_images=images,
    )


def ask_stream(
    question: str,
    image_bytes: bytes | None = None,
    image_mime: str = "image/jpeg",
    k: int = TOP_K,
) -> Iterator[str | Answer]:
    """Yields str deltas as Claude streams, then yields a final Answer object."""
    _, valid = _resources()
    retrieved = retrieve(question, k=k)
    user_prompt = _build_user_prompt(question, retrieved)
    content = _build_user_content(user_prompt, image_bytes, image_mime)

    client = Anthropic()
    full = ""
    with client.messages.stream(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        for delta in stream.text_stream:
            full += delta
            yield delta
    cited, invalid = _validate(full, valid)
    images = retrieve_images(question, image_bytes=image_bytes)
    yield Answer(
        text=full,
        sources=retrieved,
        cited_rules=cited,
        invalid_citations=invalid,
        retrieved_images=images,
    )


if __name__ == "__main__":
    import sys

    queries = sys.argv[1:] or [
        "Our team would like to put in a stronger transmission. Would this be allowed?",
        "Can I use the sale of the old transmission to offset that price?",
        "What's the weather in Tokyo today?",  # unanswerable -> should decline
    ]
    for q in queries:
        print("=" * 72)
        print(f"Q: {q}\n")
        a = ask(q)
        print(a.text)
        print(f"\n  cited:   {a.cited_rules}")
        print(f"  invalid: {a.invalid_citations}")
        print("  top sources:")
        for r in a.sources[:4]:
            print(f"    - {r.citation:26s} dist={r.distance:.3f}  | {r.text[:60].strip()}")
        print()
