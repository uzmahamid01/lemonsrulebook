# Lemons Virtual Inspector

A proof-of-concept virtual inspector for the **24 Hours of Lemons** rules.

Ask any rules question; get back a concise, **citation-faithful** answer with the source rule numbers and (when relevant) the diagrams from the rulebook.

---

## Why Challenge A

I picked the Lemons rulebook over the NYC restaurant-inspection challenge for a few reasons:

- The corpus is tractable: 4 PDFs, around 37 pages total. The NYC challenge starts with a multi-million-row dataset, and the assesstments says don't burn time on boilerplate or data wrangling.
- The interesting parts of A line up with what's actually hard about a domain assistant like this: faithful citations and visual surfacing. The assesstments calls both of those out as critical. Most of Challenge B's hour is health-code parsing plus Socrata ETL, which is less revealing of how I think about AI quality.
- It's demo-able in a one-hour pairing session. A chat UI with live citations is something to point at. A discrepancy report is mostly tables.

## What I Prototyped

The full virtual inspector has at least eight moving parts: ingest, chunk, retrieve, generate, cite, surface visuals, accept multimodal input, remember context. I focused on the two the assesstments explicitly flagged as critical:

1. **Faithful citations.** Every claim is backed by an actual rule number, and the system automatically checks that the cited rule exists in the corpus.
2. **Visual surfacing.** Diagrams from the rulebook (mostly the *How Not To Fail Tech* guide, which carries 35 figures) appear next to relevant answers.

Everything else (multi-turn memory, chat, context, eval harness, re-ranking) is left as a natural extension point. See [Extensions](#extensions-for-the-pair-session) below.

## Pipeline

```
User Question + (optional photo)
        │
        ├─────────────────────────────────────────────────────────┐
        │                                                         │
        ▼ [Text Path]                                     ▼ [Image Path]
   Voyage AI voyage-4-large                             Voyage multimodal-3
   text embedding (query mode)                         embedding (query mode)
        │                                                         │
        ▼                                                         ▼
   Chroma dense search                                    Chroma image collection
   top-6 text chunks                                      top-4 image captions
        │                                                         │
        └──────────────────────────┬────────────────----──────────┘
                                   │
                                   ▼
                    [Prompt Building]
                    Format: [Rule X.Y.Z] <text>
                    Instruction: cite with (Rule X.Y)
                                   │
                                   ▼
                    Claude Sonnet 4.6
                    temperature=0 (deterministic)
                    streaming answer
                                   │
                                   ▼
                    [Post-Hoc Validator]
                    Extract: Rule X.Y.Z from answer
                    Check: exists in corpus?
                    Flag: any hallucinated rules
                                   │
                                   ▼
              Answer + Sources + Citations + Images
```

### Key design choices

- **Rule-number-aware chunking.** The rulebook splits cleanly on headers like `4.2.1` and `3.12.5`. I chunk on those, so a chunk's ID *is* its citation (`rule_4.2.1`). That eliminates a class of citation-resolution bugs at answer time. The model can't invent a rule number that doesn't exist as a chunk we already indexed.
- **Voyage AI `voyage-4-large` embeddings.** Voyage's flagship English retrieval model at 1024 dimensions. It's rich enough to bridge indirect questions like "stronger transmission?" to "$500 vehicle budget" without needing HyDE expansion or multi-query scaffolding. An earlier version with MiniLM 384d required both of those, and v3-large worked but pulled less context per query than v4. Single embed call per query, fully deterministic. Document and query embeddings use asymmetric `input_type` for better recall.
- **Temperature 0 on Sonnet.** Same question, same answer, every time. That matters for the demo and for the determinism check in the smoke test.
- **Caption-aware image retrieval.** During ingest, each image is paired with the full text of its PDF page, which serves as the caption. The page-text approach beats a nearby-block heuristic. The visual guide's images are page-sized, so the nearest text block is just a page footer, which doesn't help retrieval at all. Each image and its caption are embedded together with `voyage-multimodal-3` to produce a single fused embedding that captures both visual and textual content, then stored in a separate Chroma collection (`lemons_images`, 33 vectors after the `MIN_IMAGE_AREA` filter drops decorative annotations like handwritten "OR" connectors). At query time, the user's question is embedded and matched against these image embeddings. Anything past the cosine-distance threshold gets dropped.
- **Strict format in the system prompt.** Citations must be `(Rule X.Y)` or `(Rule X.Y.Z)` or `(Safety Checklist p.N)` and so on. The post-hoc validator regex matches that exact format.

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.11+ | Core implementation |
| **LLM** | Claude Sonnet 4.6 (Anthropic) | Answer generation with streaming |
| **Text Embeddings** | Voyage AI `voyage-4-large` | 1024d dense retrieval (query/document asymmetric mode) |
| **Multimodal Embeddings** | Voyage AI `voyage-multimodal-3` | Image + text embedding for visual retrieval |
| **Vector Database** | Chroma | Persistent local vector store (zero-setup, metadata-aware) |
| **PDF Parsing** | PyMuPDF (fitz) | Extract text and images from PDFs with page metadata |
| **UI Framework** | Streamlit | Chat interface with streaming, images, expanders |
| **Package Manager** | `uv` | Modern Python dependency management with lock files |
| **Dependencies** | 6 packages | anthropic, chromadb, pymupdf, python-dotenv, streamlit, voyageai |

## Corpus

| Doc | Pages | Chunks | Images | Strategy |
|---|---|---|---|---|
| `prices_and_rules.pdf` | 10 | **164** rules | 1 | Regex-split on rule headers |
| `how-to-not-fail-lemons-tech-inspection.pdf` | 23 | 23 | **35** | Page-level chunks (visual-heavy) |
| `safety-checklist.pdf` | 3 | 3 | 0 | Page-level chunks |
| `24HOL_Tech-Sheet-v8_2.pdf` | 1 | 1 | 0 | Single chunk |
| **Total** | **37** | **191** | **36** | |

164 unique rule numbers got indexed. That set is the validator's known-good list.

## What Works

End-to-end smoke test (`uv run python -m src.qa`):

| Question | Cited rules | Validator | Notes |
|---|---|---|---|
| "Stronger transmission allowed?" *(assesstments Q1)* | `3.12.4.2, 3.4.2` | ✓ valid | More precise than the assesstments's reference path. Rule 3.12.4.2 literally names "transmissions" as ICE-adapted mechanical components that count toward the $500 limit. |
| "Sale of parts to offset price?" *(assesstments Q2)* | `4.7, 3.12.4.2, 4.1.2` | ✓ valid | Matches the assesstments's reference answer, plus extra context on free-parts valuation. |
| "Can I race a Tesla?" | `3.12.1, 3.12.3, 3.12.12, 3.1.1` | ✓ valid | Pre-approval gate, PPIHC EV safety rules, charging plan, and general tech inspection. Solid technical coverage. |
| "What does a proper roll cage look like?" | `3.5.1` (+ subrules) | ✓ valid | Caption-aware retrieval surfaces all six canonical roll-cage figures: *Figure 1A* (Main Hoop Type), *1B* (Halo), *1C* (Left/Right), *2* (Recommended add-ons), *3* (Front Hoop Bends), and *4* (Helmet Clearance). All from *How Not To Fail Tech* p.3-8, distances 0.38-0.51. |
| "Fuel cell installation?" | varies | ✓ valid | Surfaces both *Fuel Cell Figure 1 & 2* (p.19/20) via direct caption match, not page co-location. |
| "Weather in Tokyo?" | (none) | (no citations) | Pure refusal. No fake citations, no irrelevant pivot. |
| Validator stress test (synthetic answer with `Rule 99.99`, `Rule 7.42`, `Rule 3.2.1`) | n/a | flags 99.99, 7.42 | Validator correctly distinguishes real from hallucinated. |

Latency is 4 to 10 seconds end-to-end per question, dominated by Sonnet streaming. Voyage embedding is around 100 ms.

## Honest Limitations

1. **The structural validator can't detect miscitations.** It checks whether a rule number exists in the corpus, not whether the cited rule actually supports the specific claim being made. The fix is sketched as commented-out code in `src/qa.py`: LLM-as-judge with Haiku, one call per cited rule. See extension #1.

2. **No multi-turn memory.** Each question is standalone. The assesstments's example dialogue includes *"Can I use the sale of the old transmission to offset that price?"* where "that price" refers to a previous turn. Handling that needs conversation-aware retrieval and generation, which I didn't get to.

3. **No multi-chat or "new chat" window.** A single session lives in `st.session_state`. There's no left-sidebar list of past conversations like ChatGPT or Claude has. It would add real scaffolding for a feature that's outside the assesstments's critical path.

4. **No automated eval harness.** Smoke tests live in `src/qa.py`'s `__main__` and in the "What Works" table above. For production I'd build a hand-curated set of 30 to 50 (question, expected-citations) pairs and track precision and recall over time.

## Tradeoffs

| Choice | Why | Cost |
|---|---|---|
| Rule-number chunking vs semantic chunking | Citation IDs are chunk IDs, so there's no fuzzy citation resolution at answer time | Long rules with sub-bullets stay together; precision suffers slightly |
| Voyage `voyage-4-large` vs local MiniLM + HyDE scaffold | Single embed call per query, deterministic, simpler code (no HyDE or multi-query scaffolding), and better recall on indirect questions. v4-large pulled more relevant context per question than v3-large in my smoke tests without regressions. Also aligned with the Anthropic stack. | API dependency on Voyage. Free-tier rate limits are tight without a payment method, though 200M free tokens still apply once payment is on file. |
| Chroma vs FAISS | Metadata-aware filtering, zero-setup persistence | Slightly heavier dep |
| Streamlit vs Next.js or Gradio | Honest 1-2 hour build; pair-session friendly | Constrained UI control |
| Sonnet end-to-end vs Sonnet + Haiku router | Consistency; no routing logic to tune | A few cents more per query |
| uv vs pip | Modern, reproducible dependency management | Slightly more setup but better long term |

## How I Evaluated

No automated eval harness yet (it's in the extensions list). For the prototype I used five hand-checks:

1. **The assesstments's two example questions.** Both should produce answers that match the assesstments's reference reasoning, with a green validator badge. They do. Q1 cites Rule 3.12.4.2, which explicitly names "transmissions" as ICE-adapted parts that count toward $500. That's actually a more direct path than the assesstments's reference chain through Rule 4.2.1's exempt list. Q2 cites Rule 4.7 (Scavenger Sales) exactly as the assesstments expects.
2. **Out-of-scope rejection.** "Weather in Tokyo?" should decline without inventing rule numbers. It does.
3. **Validator integrity test.** A synthetic answer containing made-up rules like `99.99` and `7.42` should be flagged. It is.
4. **Determinism.** Same question twice should produce identical answers under `temperature=0`. It does.
5. **Visual surfacing.** Roll-cage questions should retrieve diagrams from *How Not To Fail Tech*. They do.

These five checks are wired into a single smoke test in `src/qa.py`'s `__main__`.

## Extensions for the Pair Session

Five concrete starting points, roughly ordered by how interesting they'd be per minute of pairing time:

1. **Citation-content validator.** What we can do is use a small model like Haiku to check if the cited rule actually supports the claim. This catches cases where the rule exists but doesn't support the claim being made.
2. **Conversational / multi-turn memory.** Implementing the chat history and tracking the conversation state. Before answering the user's question, we should check if there are any previous messages in the chat history and include them in the prompt as a context for the current question.
3. **Multi-chat / new chat window.** Sidebar list of past conversations, a "New chat" button, and persisting chats to disk so they survive a page refresh. 
4. **Eval harness.** Hand-curate 15 to 30 (question, expected-citations) pairs. Run them nightly. Track precision and recall of cited rules along with image-retrieval distance distributions. Wire to CI.

## Running

Requires Python 3.11+ and [`uv`](https://github.com/astral-sh/uv). Install with `brew install uv` on macOS or the [official installer](https://docs.astral.sh/uv/getting-started/installation/) on other platforms.

```bash
# 1. Install dependencies
uv sync

# 2. Set both API keys
cp .env.example .env
# then edit .env and set:
#   ANTHROPIC_API_KEY=sk-ant-...   (https://console.anthropic.com)
#   VOYAGE_API_KEY=pa-...          (https://dashboard.voyageai.com, free tier)

# 3. Ingest the PDFs (already in data/) and build the vector indexes.
# Voyage's free tier without a payment method is 3 RPM / 10K TPM, so the
# throttled build takes about 3 minutes. Adding a payment method at
# dashboard.voyageai.com unlocks higher limits, and the 200M free tokens
# still apply.
uv run python -m src.ingest    # produces chunks.json, images.json, images/ (.png)
uv run python -m src.index     # populates chroma_db/ (lemons + lemons_images collections)

# 4. Run the chat UI
uv run streamlit run app.py    # opens http://localhost:8501
```

CLI smoke test (no UI), for quick verification:

```bash
uv run python -m src.qa "Can I race a Tesla?"
uv run python -m src.qa        # runs the default 3 example questions
```

The four PDFs live in `data/` and were downloaded from the official 24 Hours of Lemons site:
- [Rulebook (Prices & Rules)](https://24hoursoflemons.com/prices-rules/)
- [How Not To Fail Lemons Tech Inspection](https://24hoursoflemons.com/wp-content/themes/lemons/assets/images/how-to-not-fail-lemons-tech-inspection.pdf)
- [Safety Checklist](https://24hoursoflemons.com/safety-checklist/)
- [Tech Sheet v8.2](https://24hoursoflemons.com/wp-content/uploads/2023/01/24HOL_Tech-Sheet-v8_1.pdf)

## Repo Structure

```
LemonsRulebook/
├── data/                       # 4 source PDFs (manually downloaded)
├── images/                     # 33 PNGs extracted from PDFs (regenerated by src.ingest;
│                               #   tiny annotation images < MIN_IMAGE_AREA are filtered out)
├── chroma_db/                  # persistent vector store (regenerated by src.index):
│                               #   • lemons          - 191 rule/page chunk embeddings
│                               #   • lemons_images   - 33 image-caption embeddings
├── chunks.json                 # ingestion artifact: chunks → Chroma `lemons` collection
├── images.json                 # ingestion artifact: image metadata + captions → `lemons_images`
├── src/
│   ├── ingest.py               # PDF → chunks + image manifest; rule-aware splitting + caption extraction
│   ├── index.py                # embed chunks + image captions → both Chroma collections
│   └── qa.py                   # retrieve → generate → validate; ask() + ask_stream() + retrieve_images()
├── app.py                      # Streamlit chat UI with streaming + citation badge + figure grid
├── .streamlit/config.toml      # Lemons-themed UI config
├── pyproject.toml              # uv-managed deps (6 packages: anthropic, chromadb, pymupdf,
│                               #   python-dotenv, streamlit, voyageai)
└── .env.example                # placeholders for ANTHROPIC_API_KEY and VOYAGE_API_KEY
```
