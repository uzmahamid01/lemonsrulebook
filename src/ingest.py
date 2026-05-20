"""Ingest the four Lemons PDFs into chunks + image manifest.

Per-doc strategy:
  - prices_and_rules.pdf      -> split on rule-number headers (e.g. "4.2.1")
  - how-to-not-fail-...pdf    -> one chunk per page (visual-heavy doc, 35 images)
  - safety-checklist.pdf      -> one chunk per page
  - 24HOL_Tech-Sheet-v8_2.pdf -> single chunk for the whole sheet

Images are extracted to images/ with page metadata so chunks can surface them.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fitz  # pymupdf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IMAGES = ROOT / "images"
CHUNKS_FILE = ROOT / "chunks.json"
IMAGES_FILE = ROOT / "images.json"

# Embedded-image filter: the PDFs store decorative annotations (handwritten
# "OR" and "AND" symbols, connecting arrows between alternative diagrams, etc.) as separate
# embedded images. The smallest real figure in the corpus is ~100K pixels²;
# annotations are 20K-40K. Dropping anything below 50K keeps every real
# diagram while filtering out the connective noise.
MIN_IMAGE_AREA = 50_000

DOC_LABELS = {
    "prices_and_rules.pdf": "Rulebook",
    "how-to-not-fail-lemons-tech-inspection.pdf": "How Not To Fail Tech",
    "safety-checklist.pdf": "Safety Checklist",
    "24HOL_Tech-Sheet-v8_2.pdf": "Tech Sheet",
}

# Matches a rule header at start of line: digits with dots, then whitespace
# (incl. nbsp via \s), then content. Examples: "1.0  WARNING", "4.2.1  Drivers".
RULE_HEADER_RE = re.compile(r"^(\d+(?:\.\d+){1,3})\s+(\S.*)$", re.MULTILINE)


@dataclass
class Chunk:
    id: str
    doc: str               # human label, e.g. "Rulebook"
    doc_file: str          # filename
    citation: str          # "Rule 4.2.1" or "Safety Checklist p.1"
    rule_number: str | None
    title: str
    text: str
    page: int              # 1-indexed; for multi-page rules, the page of the header
    image_paths: list[str] = field(default_factory=list)


@dataclass
class ImageMeta:
    """One image plus its caption (the surrounding page text, truncated).

    Caption strategy: use the normalized full page text rather than a nearby
    text-block heuristic. The "How Not To Fail Tech" guide has page-sized
    figures where the closest text block is the page footer ("Inspection 5"),
    which is useless for retrieval. The page's text describes the figure, use it as the caption.
    """
    path: str              # relative path under images/
    doc: str
    doc_file: str
    page: int
    caption: str


def _normalize(s: str) -> str:
    """Clean Unicode, collapse whitespace, and remove blank lines from text."""
    s = unicodedata.normalize("NFKC", s)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _extract_images(
    doc: fitz.Document, doc_label: str, doc_file: str, doc_stem: str
) -> tuple[dict[int, list[str]], list[ImageMeta]]:
    """Returns ({page: [paths]}, [ImageMeta with page-text captions])."""
    by_page: dict[int, list[str]] = {}
    metas: list[ImageMeta] = []
    for pno, page in enumerate(doc, start=1):
        page_text = _normalize(page.get_text())[:500]
        paths: list[str] = []
        for idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha >= 4:  # CMYK -> RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                # Drop decorative annotations ("OR", "AND", connecting arrows, etc.)
                if pix.width * pix.height < MIN_IMAGE_AREA:
                    continue
                out = IMAGES / f"{doc_stem}_p{pno}_{idx}.png"
                pix.save(out)
                rel = str(out.relative_to(ROOT))
                paths.append(rel)
                metas.append(ImageMeta(
                    path=rel, doc=doc_label, doc_file=doc_file,
                    page=pno, caption=page_text,
                ))
            except Exception as e:
                print(f"  [warn] image extract failed {doc_stem} p{pno} #{idx}: {e}")
        if paths:
            by_page[pno] = paths
    return by_page, metas


def _ingest_rulebook(doc: fitz.Document, images_by_page: dict[int, list[str]]) -> list[Chunk]:
    """Split the rulebook on rule-number headers; track which page each rule lives on."""
    page_text: list[tuple[int, str]] = []
    for pno, page in enumerate(doc, start=1):
        page_text.append((pno, _normalize(page.get_text())))

    full = ""
    page_offsets: list[tuple[int, int]] = []  # (start_offset, page_no)
    for pno, txt in page_text:
        page_offsets.append((len(full), pno))
        full += txt + "\n"

    def page_of(offset: int) -> int:
        out = 1
        for off, pno in page_offsets:
            if off <= offset:
                out = pno
            else:
                break
        return out

    matches = list(RULE_HEADER_RE.finditer(full))
    chunks: list[Chunk] = []
    for i, m in enumerate(matches):
        rule = m.group(1)
        title_line = m.group(2).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full)
        body = full[start:end].strip()
        title = title_line.split(":", 1)[0][:80]
        page = page_of(start)
        chunks.append(
            Chunk(
                id=f"rule_{rule}",
                doc="Rulebook",
                doc_file="prices_and_rules.pdf",
                citation=f"Rule {rule}",
                rule_number=rule,
                title=title,
                text=body,
                page=page,
                image_paths=images_by_page.get(page, []),
            )
        )
    return chunks


def _ingest_by_page(doc: fitz.Document, label: str, filename: str,
                    images_by_page: dict[int, list[str]]) -> list[Chunk]:
    """Create one chunk per page for visual-heavy docs (How Not To Fail, Safety Checklist)."""
    chunks: list[Chunk] = []
    for pno, page in enumerate(doc, start=1):
        text = _normalize(page.get_text())
        if not text:
            continue
        first_line = text.splitlines()[0][:80]
        chunks.append(
            Chunk(
                id=f"{filename}_p{pno}",
                doc=label,
                doc_file=filename,
                citation=f"{label} p.{pno}",
                rule_number=None,
                title=first_line,
                text=text,
                page=pno,
                image_paths=images_by_page.get(pno, []),
            )
        )
    return chunks


def _ingest_tech_sheet(doc: fitz.Document, images_by_page: dict[int, list[str]]) -> list[Chunk]:
    """Create a single chunk for the one-page Tech Sheet."""
    text = _normalize("\n".join(p.get_text() for p in doc))
    return [
        Chunk(
            id="tech_sheet",
            doc="Tech Sheet",
            doc_file="24HOL_Tech-Sheet-v8_2.pdf",
            citation="Tech Sheet",
            rule_number=None,
            title="Lemons Tech Sheet v8.2",
            text=text,
            page=1,
            image_paths=images_by_page.get(1, []),
        )
    ]


def ingest_all() -> list[Chunk]:
    """Load all PDFs from data/, extract images, chunk per-doc strategy, and persist to JSON."""
    IMAGES.mkdir(exist_ok=True)
    for f in IMAGES.glob("*.png"):
        f.unlink()

    all_chunks: list[Chunk] = []
    all_image_metas: list[ImageMeta] = []
    for filename, label in DOC_LABELS.items():
        path = DATA / filename
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        print(f"[ingest] {filename}  ({label})")
        doc = fitz.open(path)
        stem = path.stem
        images_by_page, image_metas = _extract_images(doc, label, filename, stem)
        n_images = sum(len(v) for v in images_by_page.values())
        all_image_metas.extend(image_metas)

        if filename == "prices_and_rules.pdf":
            chunks = _ingest_rulebook(doc, images_by_page)
        elif filename == "24HOL_Tech-Sheet-v8_2.pdf":
            chunks = _ingest_tech_sheet(doc, images_by_page)
        else:
            chunks = _ingest_by_page(doc, label, filename, images_by_page)

        print(f"         pages={len(doc)}  chunks={len(chunks)}  images={n_images}")
        all_chunks.extend(chunks)
        doc.close()

    CHUNKS_FILE.write_text(json.dumps([asdict(c) for c in all_chunks], indent=2))
    print(f"\n[done] {len(all_chunks)} total chunks -> {CHUNKS_FILE.relative_to(ROOT)}")

    IMAGES_FILE.write_text(json.dumps([asdict(m) for m in all_image_metas], indent=2))
    n_capped = sum(1 for m in all_image_metas if m.caption)
    print(f"[done] {len(all_image_metas)} images ({n_capped} with captions) -> {IMAGES_FILE.relative_to(ROOT)}")

    rule_ids = {c.rule_number for c in all_chunks if c.rule_number}
    for required in ("4.2.1", "4.7"):
        flag = "OK" if required in rule_ids else "MISSING"
        print(f"   [{flag}] Rule {required}")

    return all_chunks


if __name__ == "__main__":
    ingest_all()
