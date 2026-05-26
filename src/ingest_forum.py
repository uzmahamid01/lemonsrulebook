"""Ingest Lemons forum threads into a simplified Q&A document set.

Reads the two JSONL exports under lemons_forums/, groups posts by thread,
strips HTML, and asks Claude Haiku 4.5 to summarize each thread into one
plain-English Q&A entry. The Q&A text is what later gets embedded; the
original posts + thread URL are kept as metadata for grounding.

Resumability: every successful Claude call appends a line to
forum_summaries_cache.jsonl, so a crash costs only seconds of progress.
Skipped threads (no rules-relevant content) are also cached so re-runs
don't re-ask Claude.

Run:
    uv run python -m src.ingest_forum --limit 20   # smoke test
    uv run python -m src.ingest_forum              # full pipeline
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv

from src.ingest import _normalize

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
FORUM_DIR = ROOT / "lemons_forums"
FORUM_FILES = [
    FORUM_DIR / "lemons_tech.json",
    FORUM_DIR / "lemons_newcomers.json",
]
CHUNKS_FILE = ROOT / "forum_chunks.json"
CACHE_FILE = ROOT / "forum_summaries_cache.jsonl"

SUMMARY_MODEL = "claude-haiku-4-5"
SUMMARY_MAX_TOKENS = 500
SUMMARY_SLEEP_SEC = 0.5  # gentle throttle between Claude calls

MIN_POSTS = 2
MIN_CLEAN_TEXT_CHARS = 200
THREAD_TEXT_BUDGET = 12_000      # cap full thread text before sending to Claude
THREAD_TEXT_HEAD = 9_000          # first N chars kept
THREAD_TEXT_TAIL = 3_000          # last N chars kept (preserves resolution)
ORIGINAL_TEXT_TRUNCATE = 4_000    # how much cleaned text we stash in metadata

SKIP_TITLE_RE = re.compile(r"^(test|delete|spam)\b", re.IGNORECASE)
SIG_DIV_RE = re.compile(r'<div class="sig-content">.*?</div>', re.DOTALL)
QUOTEBOX_DIV_RE = re.compile(r'<div class="quotebox">.*?</div>', re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")

SUMMARY_SYSTEM_PROMPT = """\
You condense a 24 Hours of Lemons forum thread into a single Q&A entry for a
rules-assistant retrieval index.

Return STRICT JSON with exactly two string fields:
{"question": "...", "answer": "..."}

Rules:
- "question": one sentence, in the voice of a team asking what this thread is
  really about (e.g. "Can I run an aftermarket fuel cell?").
- "answer": 2-5 sentences of the plain-English consensus from the thread.
  Translate forum jargon into everyday language a first-time team would
  understand.
- Do NOT invent rule numbers. Only mention a rule number (like "Rule 4.2.1")
  if a post in the thread explicitly cites that exact rule.
- If the thread has no rules-relevant content (off-topic banter, spam, build
  diaries with no rules question), return {"question": "", "answer": ""}.
- Output ONLY the JSON object. No prose before or after, no markdown fences.
"""


@dataclass
class ForumChunk:
    id: str
    doc: str
    doc_file: str
    citation: str
    rule_number: str | None
    title: str
    text: str
    page: int
    image_paths: list[str] = field(default_factory=list)
    thread_url: str = ""
    thread_id: str = ""
    author: str = ""
    creation_time: str = ""
    source_file: str = ""
    original_post_count: int = 0
    original_text_truncated: str = ""


# ---------------------------------------------------------------------------
# Thread reconstruction
# ---------------------------------------------------------------------------

def load_threads(path: Path) -> dict[str, dict]:
    """Parse a forum JSONL export into {thread_id: {title, url, author, creation_time, posts: [...]}}."""
    threads: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            t = item.get("type")
            p = item.get("path") or []
            if t == "thread" and len(p) >= 2:
                tid = p[1]
                threads.setdefault(tid, {
                    "thread_id": tid,
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "author": "",
                    "creation_time": "",
                    "posts": [],
                })
                threads[tid]["title"] = item.get("title", threads[tid]["title"])
                threads[tid]["url"] = item.get("url", threads[tid]["url"])
            elif t == "post" and len(p) >= 2:
                tid = p[1]
                bucket = threads.setdefault(tid, {
                    "thread_id": tid,
                    "title": "",
                    "url": "",
                    "author": "",
                    "creation_time": "",
                    "posts": [],
                })
                bucket["posts"].append({
                    "author": item.get("author", ""),
                    "creation_time": item.get("creation_time", ""),
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                })
    return threads


def _clean_html(s: str) -> str:
    """Strip HTML signature blocks, quote boxes, then all tags, then NFKC-normalize."""
    s = SIG_DIV_RE.sub(" ", s)
    s = QUOTEBOX_DIV_RE.sub(" ", s)
    s = TAG_RE.sub(" ", s)
    s = html.unescape(s)
    return _normalize(s)


def _filter_threads(threads: dict[str, dict]) -> dict[str, dict]:
    """Drop threads with too few posts, too little content, or trivial titles."""
    kept: dict[str, dict] = {}
    for tid, th in threads.items():
        posts = th.get("posts", [])
        if len(posts) < MIN_POSTS:
            continue
        title = th.get("title", "")
        if title and SKIP_TITLE_RE.search(title):
            continue
        posts.sort(key=lambda p: p.get("creation_time", ""))
        cleaned = [{**p, "clean": _clean_html(p.get("content", ""))} for p in posts]
        total = sum(len(p["clean"]) for p in cleaned)
        if total < MIN_CLEAN_TEXT_CHARS:
            continue
        th = dict(th)
        th["posts"] = cleaned
        if cleaned:
            th["author"] = th.get("author") or cleaned[0].get("author", "")
            th["creation_time"] = th.get("creation_time") or cleaned[0].get("creation_time", "")
        kept[tid] = th
    return kept


def _build_thread_text(thread: dict) -> str:
    """Concatenate posts into one string for the LLM; truncate head+tail to stay under budget."""
    parts = []
    for p in thread["posts"]:
        author = p.get("author", "?")
        ts = p.get("creation_time", "")
        body = p.get("clean", "")
        parts.append(f"[{author} @ {ts}]\n{body}")
    full = "\n\n".join(parts)
    if len(full) <= THREAD_TEXT_BUDGET:
        return full
    return full[:THREAD_TEXT_HEAD] + "\n\n[...truncated...]\n\n" + full[-THREAD_TEXT_TAIL:]


# ---------------------------------------------------------------------------
# Claude summarization (resumable)
# ---------------------------------------------------------------------------

def _load_cache() -> dict[str, dict]:
    """Read the append-only cache into {thread_id: result_dict}."""
    if not CACHE_FILE.exists():
        return {}
    done: dict[str, dict] = {}
    with CACHE_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = item.get("thread_id")
            if tid:
                done[tid] = item
    return done


def _append_cache(entry: dict) -> None:
    """Append one JSON line to the cache and flush so a crash preserves work."""
    with CACHE_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()


def _parse_summary_json(raw: str) -> tuple[str, str]:
    """Pull {"question","answer"} out of Claude's output, tolerating stray prose."""
    raw = raw.strip()
    # Try direct parse; fall back to finding the first {...} block.
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return "", ""
        try:
            obj = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return "", ""
    q = (obj.get("question") or "").strip()
    a = (obj.get("answer") or "").strip()
    return q, a


def summarize_thread(client: Anthropic, thread_text: str, title: str) -> tuple[str, str]:
    """Call Claude Haiku for a Q&A summary. Returns (question, answer); ("","") means skip."""
    user_content = (
        f"Thread title: {title}\n\n"
        f"Thread posts (in chronological order):\n{thread_text}"
    )
    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=SUMMARY_MODEL,
                max_tokens=SUMMARY_MAX_TOKENS,
                temperature=0,
                system=SUMMARY_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            text = "".join(b.text for b in msg.content if b.type == "text")
            return _parse_summary_json(text)
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"    rate-limited, sleeping {wait}s then retrying")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if 500 <= getattr(e, "status_code", 0) < 600:
                wait = 10 * (attempt + 1)
                print(f"    server error {e.status_code}, sleeping {wait}s then retrying")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Claude summarization failed after 3 attempts")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _make_chunk(thread: dict, source_file: str, question: str, answer: str) -> ForumChunk:
    title = thread.get("title", "") or "(untitled)"
    citation_title = title[:50]
    cleaned_concat = "\n\n".join(p.get("clean", "") for p in thread["posts"])
    return ForumChunk(
        id=f"forum_thread_{thread['thread_id']}",
        doc="Forum",
        doc_file=source_file,
        citation=f"Forum: {citation_title}",
        rule_number=None,
        title=title,
        text=f"Q: {question}\nA: {answer}",
        page=0,
        image_paths=[],
        thread_url=thread.get("url", ""),
        thread_id=thread["thread_id"],
        author=thread.get("author", ""),
        creation_time=thread.get("creation_time", ""),
        source_file=source_file,
        original_post_count=len(thread["posts"]),
        original_text_truncated=cleaned_concat[:ORIGINAL_TEXT_TRUNCATE],
    )


def run(limit: int | None = None, only_file: str | None = None) -> list[ForumChunk]:
    """Load + filter threads from both forum JSONLs, summarize each, write forum_chunks.json."""
    cache = _load_cache()
    print(f"[forum] cache has {len(cache)} prior results")

    client = Anthropic()
    chunks: list[ForumChunk] = []
    processed = 0

    for src_path in FORUM_FILES:
        if only_file and src_path.name != only_file:
            continue
        if not src_path.exists():
            print(f"[skip] {src_path} not found")
            continue
        print(f"[forum] loading {src_path.name}")
        threads = load_threads(src_path)
        threads = _filter_threads(threads)
        print(f"[forum] {src_path.name}: {len(threads)} threads after filtering")

        for tid, thread in threads.items():
            if limit is not None and processed >= limit:
                break

            cached = cache.get(tid)
            if cached is not None:
                if cached.get("skipped"):
                    processed += 1
                    continue
                question = cached.get("question", "")
                answer = cached.get("answer", "")
                if question and answer:
                    chunks.append(_make_chunk(thread, src_path.name, question, answer))
                processed += 1
                continue

            thread_text = _build_thread_text(thread)
            try:
                question, answer = summarize_thread(client, thread_text, thread.get("title", ""))
            except Exception as e:
                print(f"  [error] thread {tid}: {e}")
                # Do not cache hard failures; allow re-run to retry.
                processed += 1
                time.sleep(SUMMARY_SLEEP_SEC)
                continue

            entry = {"thread_id": tid, "source_file": src_path.name}
            if not question or not answer:
                entry["skipped"] = True
                _append_cache(entry)
            else:
                entry["question"] = question
                entry["answer"] = answer
                _append_cache(entry)
                chunks.append(_make_chunk(thread, src_path.name, question, answer))

            processed += 1
            if processed % 25 == 0:
                print(f"  [progress] processed={processed} kept={len(chunks)}")
            time.sleep(SUMMARY_SLEEP_SEC)

        if limit is not None and processed >= limit:
            break

    CHUNKS_FILE.write_text(json.dumps([asdict(c) for c in chunks], indent=2))
    print(f"[done] {len(chunks)} forum Q&A chunks -> {CHUNKS_FILE.relative_to(ROOT)}")
    print(f"[done] cache at {CACHE_FILE.relative_to(ROOT)}")
    return chunks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize Lemons forum threads into Q&A chunks.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N threads total (smoke test).")
    ap.add_argument("--only", type=str, default=None,
                    help="Only process this filename (e.g. lemons_tech.json).")
    args = ap.parse_args(argv)
    run(limit=args.limit, only_file=args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
