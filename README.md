# Resume Job Matcher AI

> **AI-powered semantic job matching** — upload your resume, get ranked remote job matches with fit scores, skills gap analysis, and actionable recommendations. Built on a RAG pipeline with local vector embeddings and LLM scoring.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-faiss--cpu-00A67E)](https://github.com/facebookresearch/faiss)
[![Together.ai](https://img.shields.io/badge/Together.ai-LLM-7C3AED)](https://together.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Problem Statement

Job searching is a manual, exhausting process. You scroll through hundreds of postings, copy-paste your resume into application forms, and have no clear signal about which roles actually fit your skills — until after you've applied and been rejected.

**Resume Job Matcher AI solves this in three steps:**

1. **Match** — Upload your resume. A local embedding model encodes it as a semantic vector and searches 300+ live remote job postings using FAISS cosine similarity — finding roles that *mean* something similar to your background, not just roles that share the same keywords.
2. **Score** — A Llama-3.2-3B LLM reads your resume alongside each matched job description and produces a structured 0–100 fit score with matched skills, missing skills, and specific recommendations.
3. **Act** — Results are displayed in a ranked, colour-coded Streamlit UI with one-click apply links and full skills gap analysis for each role.

---

## Key Features

- **Semantic search** — FAISS cosine similarity matches meaning, not keywords. "ML engineer" matches "machine learning developer" automatically
- **Local embeddings** — sentence-transformers runs entirely on CPU, free, offline, with no API dependency
- **LLM-powered scoring** — Llama-3.2-3B via Together.ai generates structured fit scores, skills gaps, and recommendations for each match
- **Config-driven architecture** — zero hardcoded model names, prompts, or limits in Python; all values read from `config/app_config.json` at runtime
- **Switchable embedding models** — sidebar button queries Together.ai live for available serverless models; one-click switch with config update
- **Role filtering** — pre-defined role keyword groups (ML Engineer, LLM Engineer, DevOps, etc.) bias FAISS retrieval toward preferred roles
- **PDF resume support** — pypdf extracts text from uploaded PDF resumes directly in the browser

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                        │
│                      app.py  (port 8501)                     │
│                                                              │
│  Role selector  │  K slider  │  PDF upload  │  Run button    │
│                                                              │
│              Upload resume → embed → search → score          │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    LOCAL EMBED LAYER                         │
│            sentence-transformers  (CPU, offline)             │
│                                                              │
│  ┌──────────────────┐        ┌──────────────────────────┐   │
│  │  embed_jobs.py   │        │     match_jobs.py        │   │
│  │  (pipeline step) │        │  (per-request, runtime)  │   │
│  │  encodes all     │        │  encodes resume → vector  │   │
│  │  job descriptions│        │  normalises to length 1   │   │
│  └────────┬─────────┘        └─────────────┬────────────┘   │
└───────────┼──────────────────────────────── ┼ ───────────────┘
            │  job_vectors.npy                │  resume_vec
            ▼                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    FAISS INDEX                               │
│           IndexFlatIP  ·  384-dim  ·  cosine similarity      │
│                                                              │
│  build_faiss_index.py ──▶ faiss.index  (binary file)        │
│                                                              │
│  index.search(resume_vec, top_k) ──▶ scores + indices       │
└──────────────────────────┬───────────────────────────────────┘
                           │  top-K matched job texts
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  TOGETHER.AI LLM                             │
│           Llama-3.2-3B-Instruct-Turbo  (serverless)          │
│                                                              │
│  Prompt: resume_text + job_text (from config/prompts/)       │
│  temperature=0.0  ·  max_tokens=220                          │
│                                                              │
│  Output:  FIT_SCORE: 72                                      │
│           MATCHED_SKILLS: Python, Docker, AWS                │
│           MISSING_SKILLS: Kubernetes, Terraform              │
│           RECOMMENDATIONS: Get CKA certified, ...            │
│           SUMMARY: Strong cloud background but ...           │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   RESULTS UI                                 │
│                                                              │
│  Ranked job cards  ·  colour-coded scores  ·  skill pills    │
│  skills gap  ·  recommendations  ·  apply links              │
└─────────────────────────────────────────────────────────────┘
```

---

## AI Pipeline

### Step 1 — Fetch & Clean
**Script:** `src/fetch_jobs.py` → `src/clean_jobs.py`
**Input:** Remotive public API
**Output:** `data/jobs/jobs_clean.json`
**What it does:** Pulls 300+ remote job listings, strips HTML tags, removes boilerplate patterns (EEO disclaimers, legal text) defined in `config/app_config.json`

### Step 2 — Embed Jobs
**Script:** `src/embed_jobs.py`
**Input:** `jobs_clean.json`
**Output:** `data/index/job_vectors.npy`
**What it does:** Encodes every job description into a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2` locally. Runs once; rebuild only when jobs change.

### Step 3 — Build FAISS Index
**Script:** `src/build_faiss_index.py`
**Input:** `job_vectors.npy`
**Output:** `data/index/faiss.index` + `data/index/job_meta.json`
**What it does:** Loads all vectors, normalises to unit length, adds to a FAISS `IndexFlatIP` (exact inner product = cosine similarity on normalised vectors)

### Step 4 — Match & Score *(runtime, per request)*
**Scripts:** `src/match_jobs.py` → `src/score_explain.py`
**Input:** Uploaded resume (PDF or TXT)
**Output:** Ranked scored results displayed in UI
**What it does:** Embeds resume → FAISS search → top-K job texts injected into LLM prompt → structured fit score parsed from LLM output

---

## Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Frontend | Streamlit | 1.31+ | Web UI — upload, sliders, results, model switcher |
| Embedding | sentence-transformers | 3.x | Local CPU semantic embeddings (all-MiniLM-L6-v2) |
| Vector Search | FAISS | faiss-cpu 1.8+ | Cosine similarity search over job vectors |
| LLM | Together.ai Llama-3.2-3B | Instruct-Turbo | Structured fit scoring and explanation |
| Job Data | Remotive API | — | 300+ live remote job listings, no key required |
| Resume Parsing | pypdf | 4.x | Extract text from uploaded PDF resumes |
| Config | JSON + .txt prompts | — | Runtime config — zero hardcoded values in Python |
| Env | python-dotenv | 1.x | Load `TOGETHER_API_KEY` from `.env` locally |

---

## Project Structure

```
resume_job_matcher/
│
├── app.py                        # Streamlit UI — main entry point
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── app_config.json           # All settings: models, limits, roles, noise patterns
│   └── prompts/
│       └── score_job_v2.txt      # LLM prompt template (no hardcoding in Python)
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Config loader + helper functions
│   ├── embedder.py               # Unified local/API embedding abstraction
│   ├── fetch_jobs.py             # Pulls jobs from Remotive API
│   ├── clean_jobs.py             # HTML stripper + noise pattern remover
│   ├── embed_jobs.py             # Batch embeds all clean jobs
│   ├── build_faiss_index.py      # Builds FAISS index from job vectors
│   ├── match_jobs.py             # Embeds resume → FAISS search → ranked matches
│   ├── score_explain.py          # LLM scoring + structured KV output parser
│   └── model_search.py           # Together.ai model discovery + config updater
│
└── data/                         # Generated at runtime — gitignored
    ├── jobs/
    │   ├── jobs_raw.json
    │   └── jobs_clean.json
    ├── index/
    │   ├── job_vectors.npy
    │   ├── faiss.index
    │   └── job_meta.json
    └── resume/                   # Optional: drop resume PDF here
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- A Together.ai API key — [get one free at together.ai](https://together.ai)

### 1. Clone and configure

```powershell
git clone https://github.com/kaizermm/resume_job_matcher.git
cd resume_job_matcher
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
pip install sentence-transformers
```

### 3. Set API key

```powershell
# PowerShell — current session
$env:TOGETHER_API_KEY = "your_key_here"

# PowerShell — permanent
setx TOGETHER_API_KEY "your_key_here"
```

### 4. Run the data pipeline

```powershell
python -m src.fetch_jobs            # Fetch jobs from Remotive API
python -m src.clean_jobs            # Clean and normalise job text
python -m src.embed_jobs            # Embed jobs as vectors (~90MB model download on first run)
python -m src.build_faiss_index     # Build FAISS index
```

### 5. Launch the app

```powershell
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), upload your resume PDF, select a role, and click **Find My Best Jobs**.

### 6. Verify

```powershell
python -c "import faiss, json; idx=faiss.read_index('data/index/faiss.index'); print('Vectors:', idx.ntotal)"
# Expected: Vectors: 22  (or however many jobs were fetched)
```

---

## Environment Variables

```bash
# .env  (copy from .env.example)
TOGETHER_API_KEY=your_key_here
```

---

## API Reference

The app is a self-contained Streamlit application. The internal pipeline functions are:

| Function | Module | Description |
|----------|--------|-------------|
| `embed_texts(texts, model, client)` | `src/embedder.py` | Encode text list → numpy vectors (local or API) |
| `embed_one(text, model, client)` | `src/embedder.py` | Encode single text → numpy vector |
| `load_faiss_index()` | `src/match_jobs.py` | Load index, meta, and clean jobs from disk |
| `match_resume_to_jobs(resume, index, ...)` | `src/match_jobs.py` | Embed resume → FAISS search → ranked matches |
| `score_top_jobs(resume, matches, client)` | `src/score_explain.py` | LLM score each match → structured result dicts |
| `parse_kv_output(raw, fields)` | `src/score_explain.py` | Parse LLM KEY:VALUE text → Python dict |
| `get_available_embedding_models(api_key)` | `src/model_search.py` | Query Together.ai for live serverless models |
| `set_embed_model(model_id)` | `src/model_search.py` | Update embed model in app_config.json |

---

## Design Decisions

**Why local embeddings instead of Together.ai?**
Together.ai removed three different embedding models from their free serverless tier during development (`BAAI/bge-base-en-v1.5`, `m2-bert-80M-8k-retrieval`, `WhereIsAI/UAE-Large-V1`), each causing `model_not_available` errors. Moving to `sentence-transformers` running locally gave full control — free, offline, stable, and never subject to provider outages.

**Why FAISS IndexFlatIP instead of approximate indexes?**
With 300 jobs, an exact brute-force inner product search is fast enough (milliseconds) and gives perfect recall. Approximate indexes (IVFFlat, HNSW) trade accuracy for speed at millions of vectors — unnecessary at this scale and adds tuning complexity with no benefit.

**Why config-driven architecture?**
Every time a model name, prompt, or limit needed to change, only `app_config.json` or a `.txt` file was edited — no Python files touched. This proved its value repeatedly: three embedding model switches and two prompt iterations, all handled without touching application code.

**Why temperature=0.0 for LLM scoring?**
The parser (`parse_kv_output`) expects a strict `KEY:VALUE` format. Any randomness risks generating markdown, reordering fields, or adding prose — all of which break the parser. Deterministic output at temperature 0 makes the structured format reliable.

**Why `python -m src.module` instead of `python src/module.py`?**
Running scripts directly makes Python treat `src/` as the working directory, breaking all relative imports (`from src.config import ...`). The `-m` flag runs the file as a module within the project root, keeping all import paths correct.

---

## Roadmap

- [ ] Reranking with `Salesforce/Llama-Rank-V1` for improved result ordering
- [ ] CSV / JSON export of scored results
- [ ] Skill gap radar chart visualisation
- [ ] Location and salary range filters
- [ ] Streamlit Cloud one-click deployment
- [ ] Docker containerisation
- [ ] Job source expansion beyond Remotive (LinkedIn, Greenhouse, Lever)
- [ ] Feedback loop — mark good/bad matches to improve future results

---
