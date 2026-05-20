"""Streamlit chat UI for the Lemons Virtual Inspector.

Handles the interactive chat interface: displays message history, accepts questions
(with optional photo uploads), streams answers from the LLM, and renders citation badges
plus related figures from the rulebook. Uses Streamlit's session state to track conversation.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.qa import Answer, ask_stream

load_dotenv()
ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Lemons Virtual Inspector",
    page_icon="🏁",
    layout="wide",
)

st.title("🏁 Lemons Virtual Inspector")
st.caption(
    "Ask anything about the 24 Hours of Lemons rules. "
    "Citations are auto-validated against the rulebook."
)

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error(
        "`ANTHROPIC_API_KEY` not set. Add it to `.env` and restart `streamlit run app.py`."
    )
    st.stop()

if not (ROOT / "chroma_db").exists():
    st.error(
        "Vector index not found. Run `uv run python -m src.ingest && uv run python -m src.index` first."
    )
    st.stop()


def _render_extras(a: Answer) -> None:
    """Citation badge, related figures (from cited chunks), and Sources expander."""
    # Caption-aware image retrieval. Figures are matched against the *question*
    # text via caption similarity, not page co-location with cited chunks.
    # That means a roll-cage answer surfaces the roll-cage diagram even when
    # other diagrams share its page.
    if a.retrieved_images:
        st.markdown(
            "**Related figures** *(matched by caption similarity to the question)*"
        )
        n_cols = min(3, len(a.retrieved_images))
        cols = st.columns(n_cols)
        for i, img in enumerate(a.retrieved_images[:6]):
            full = ROOT / img["path"]
            if not full.exists():
                continue
            # Pick the first caption line that isn't the doc header or page-number footer.
            lines = [
                ln.strip() for ln in img.get("caption", "").splitlines() if ln.strip()
            ]
            informative = [
                ln
                for ln in lines
                if not ln.isdigit()
                and "Tech Inspection" not in ln
                and "PRICES" not in ln
            ]
            label_text = (
                informative[0] if informative else (lines[0] if lines else "")
            )[:60]
            # Citation format: Doc p.Page — description (similarity score)
            citation = f"**{img['doc']} p.{img['page']}**"
            caption = f"{citation} — {label_text}\n*(match: {img['distance']:.2f})*"
            cols[i % n_cols].image(str(full), caption=caption, use_container_width=True)

    if a.invalid_citations:
        bad = ", ".join(c for c in a.invalid_citations)
        st.error(f"⚠ Unverified citation(s): {bad} — not found in the corpus.")
    elif a.cited_rules:
        n = len(a.cited_rules)
        st.success(
            f"✓ All {n} citation{'s' if n != 1 else ''} valid against the rulebook."
        )
    else:
        st.info("No rule citations in this answer.")

    with st.expander(f"Sources ({len(a.sources)} retrieved)", expanded=False):
        for r in a.sources:
            st.markdown(
                f"**{r.citation}** &nbsp; *(distance {r.distance:.3f} · {r.doc} p.{r.page})*"
            )
            preview = r.text[:400] + ("…" if len(r.text) > 400 else "")
            st.markdown(preview)
            st.divider()


# --- chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "user" and msg.get("image_bytes"):
            st.image(msg["image_bytes"], width=320)
        if msg["role"] == "assistant" and msg.get("answer"):
            _render_extras(msg["answer"])

# --- input row: chat input with optional inline photo attachment ---
prompt = st.chat_input(
    "Ask about Lemons rules… (attach a photo optionally)",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "webp"],
)

if prompt and prompt.text:
    q = prompt.text
    uploaded = prompt["files"][0] if prompt["files"] else None
    image_bytes = uploaded.getvalue() if uploaded else None
    image_mime = uploaded.type if uploaded else "image/jpeg"

    st.session_state.messages.append(
        {"role": "user", "content": q, "image_bytes": image_bytes}
    )
    with st.chat_message("user"):
        st.markdown(q)
        if image_bytes:
            st.image(image_bytes, width=320)

    with st.chat_message("assistant"):
        text_placeholder = st.empty()
        with st.spinner("Consulting the rulebook…"):
            text_so_far = ""
            final: Answer | None = None
            try:
                for item in ask_stream(
                    q, image_bytes=image_bytes, image_mime=image_mime
                ):
                    if isinstance(item, str):
                        text_so_far += item
                        text_placeholder.markdown(text_so_far + "▌")
                    else:
                        final = item
            except Exception as e:
                st.error(f"Error from the inspector: {e}")
                final = None
        text_placeholder.markdown(text_so_far)
        if final is not None:
            _render_extras(final)

    st.session_state.messages.append(
        {"role": "assistant", "content": text_so_far, "answer": final}
    )
    st.rerun()

# sidebar
with st.sidebar:
    st.subheader("About")
    st.markdown(
        "This is a proof-of-concept built for the GovStream.ai coding challenge.\n\n"
        "**Pipeline:**\n"
        "- Text: Voyage AI `voyage-4-large` → Chroma top-6 rules\n"
        "- Images: Voyage multimodal-3 → Chroma embeddings for image and image captions\n"
        "- Generate: Claude Sonnet 4.6 (temp=0, strict citation format)\n"
        "- Validate: Post-hoc citation checker (Rule X.Y, Doc p.N, etc.)\n\n"
        "**Corpus:** Rulebook (164 rules), How Not To Fail Tech (35 figures), "
        "Safety Checklist, Tech Sheet."
    )
    st.markdown("**Try these:**")
    for ex in [
        "Our team would like to put in a stronger transmission. Would this be allowed?",
        "Can I use the sale of the old transmission to offset that price?",
        "What head and neck restraints are required?",
        "What are the rules around roll cages?",
        "Can I race a Tesla?",
    ]:
        st.markdown(f"- {ex}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
