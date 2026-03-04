# 🎯 Resume Job Matcher AI

> AI-powered semantic job matching — upload your resume, get ranked matches with fit scores, skills gap analysis, and actionable recommendations.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green?style=flat-square)
![Together.ai](https://img.shields.io/badge/Together.ai-LLM-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📚 Table of Contents

- [How It Works](#-how-it-works)
- [Architecture Diagram](#-architecture-diagram)
- [Flow Diagram](#-flow-diagram)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Configuration Reference](#-configuration-reference)
- [Cost Estimate](#-cost-estimate)
- [Files to Remove](#-files-to-remove)

---

## 🧠 How It Works

The system runs in two phases:

**Phase 1 — Build (run once):** Fetch remote jobs → clean text → embed as vectors → store in FAISS index.

**Phase 2 — Match (every run):** Upload resume → embed resume → cosine search FAISS → score top-K jobs with LLM → display results.

Zero hardcoded prompts or model names in Python. Everything is driven by `config/app_config.json` and `config/prompts/*.txt`.

---

## 🏗 Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        RESUME JOB MATCHER                                │
├──────────────────┬───────────────────────┬───────────────────────────────┤
│  DATA PIPELINE   │    VECTOR LAYER        │       APP LAYER               │
│  (run once)      │                        │       (every request)         │
├──────────────────┼───────────────────────┼───────────────────────────────┤
│                  │                        │                               │
│  🌐 Remotive API │                        │  👤 User uploads Resume       │
│  fetch_jobs.py   │                        │  (PDF or TXT)                 │
│       │          │                        │         │                     │
│       ▼          │                        │         ▼                     │
│  🧹 clean_jobs   │                        │  🔢 Local Embedder            │
│  Strip HTML      │                        │  sentence-transformers        │
│  Remove noise    │                        │  all-MiniLM-L6-v2             │
│       │          │                        │         │                     │
│       ▼          │                        │         ▼                     │
│  🔢 embed_jobs   │   ┌────────────────┐   │  🗄 FAISS Search              │
│  sentence-       │──▶│  FAISS Index   │◀──│  top-K nearest jobs           │
│  transformers    │   │  IndexFlatIP   │   │         │                     │
│       │          │   │  384-dim vecs  │   │         ▼                     │
│       ▼          │   │  cosine sim    │   │  🤖 Together.ai LLM           │
│  💾 job_vectors  │   └────────────────┘   │  Llama-3.2-3B-Turbo           │
│  .npy + .index   │                        │  Fit score + explanation      │
│                  │                        │         │                     │
│                  │                        │         ▼                     │
│                  │                        │  🎯 Streamlit Results UI      │
│                  │                        │  Score · Skills · Recs        │
└──────────────────┴───────────────────────┴───────────────────────────────┘
```

---

## 🔄 Flow Diagram

```
User uploads Resume (PDF / TXT)
              │
              ▼
┌─────────────────────────────────┐
│        STREAMLIT FRONTEND       │
│  Role selector · K slider       │
│  File uploader · Run button     │
└──────────────┬──────────────────┘
               │
    ┌──────────▼───────────┐
    │   LOCAL EMBEDDER     │
    │  sentence-transformers│
    │  all-MiniLM-L6-v2    │
    │  (free, runs on CPU)  │
    └──────────┬───────────┘
               │  resume vector (384-dim)
    ┌──────────▼───────────┐
    │     FAISS INDEX      │
    │   IndexFlatIP        │
    │   cosine similarity  │       ◀── Pre-built from job embeddings
    │   returns top-K jobs │
    └──────────┬───────────┘
               │  top-K job texts + metadata
    ┌──────────▼───────────┐
    │   TOGETHER.AI LLM    │
    │  Llama-3.2-3B-Turbo  │
    │  Structured prompt   │       ◀── Prompt from config/prompts/score_job_v2.txt
    │  FIT_SCORE: 0-100    │
    │  MATCHED_SKILLS: ... │
    │  MISSING_SKILLS: ... │
    │  RECOMMENDATIONS: ...│
    │  SUMMARY: ...        │
    └──────────┬───────────┘
               │
    ┌──────────▼───────────┐
    │    RESULTS UI        │
    │  Ranked job cards    │
    │  Colour-coded scores │
    │  Skills gap pills    │
    │  Apply links         │
    └──────────────────────┘
```

---

## 🛠 Technology Stack

| Layer | Technology | Purpose | Cost |
|-------|-----------|---------|------|
| **Frontend** | Streamlit 1.31+ | Web UI — file upload, sliders, results display, model switcher | Free |
| **Embedding** | sentence-transformers `all-MiniLM-L6-v2` | Local CPU vector embeddings for resume and jobs | Free (local) |
| **Vector Search** | FAISS `faiss-cpu` | Cosine similarity search across all job vectors | Free |
| **LLM Scoring** | Together.ai `Llama-3.2-3B-Instruct-Turbo` | Structured fit score, skills gap, recommendations | ~$0.01/run |
| **Job Data** | Remotive API | Fetches 300+ live remote job listings (no key needed) | Free |
| **Model Search** | Together.ai `/v1/models` | Sidebar button to discover available serverless models | Free |
| **Resume Parsing** | pypdf | Extracts text from uploaded PDF resumes | Free |
| **Config** | JSON + `.txt` prompts | Zero hardcoded strings — fully config-driven at runtime | — |
| **Env** | python-dotenv | Load `TOGETHER_API_KEY` from `.env` file locally | Free |

---

## 📁 Project Structure

```
resume_job_matcher/
│
├── app.py                        ← Streamlit web UI (main entry point)
├── write_files.py                ← Regenerates all src/ files from scratch
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── app_config.json           ← All settings: models, limits, roles, noise patterns
│   └── prompts/
│       └── score_job_v2.txt      ← LLM prompt template (no Python hardcoding)
│
├── src/
│   ├── __init__.py
│   ├── config.py                 ← Config loader + 10 helper functions
│   ├── embedder.py               ← Unified local/API embedding abstraction
│   ├── fetch_jobs.py             ← Pulls jobs from Remotive API
│   ├── clean_jobs.py             ← HTML stripper + noise pattern remover
│   ├── embed_jobs.py             ← Batch embeds all clean jobs
│   ├── build_faiss_index.py      ← Builds FAISS index from job vectors
│   ├── match_jobs.py             ← Embeds resume → FAISS search → ranked matches
│   ├── score_explain.py          ← LLM scoring + KV output parser
│   └── model_search.py           ← Together.ai model discovery + config updater
│
└── data/                         ← Generated at runtime (gitignored)
    ├── jobs/
    │   ├── jobs_raw.json         ← Raw Remotive API response
    │   └── jobs_clean.json       ← Cleaned job descriptions
    ├── index/
    │   ├── job_vectors.npy       ← Embedding matrix
    │   ├── faiss.index           ← FAISS binary index
    │   └── job_meta.json         ← Job metadata (title, company, url, tags)
    └── resume/                   ← Drop your resume PDF here (optional)
```

---

## ⚡ Quick Start

### 1. Install dependencies

```powershell
pip install -r requirements.txt
pip install sentence-transformers
```

### 2. Set your Together.ai API key

```powershell
# PowerShell
$env:TOGETHER_API_KEY = "your_key_here"

# Or permanently
setx TOGETHER_API_KEY "your_key_here"
```

### 3. Write all source files

```powershell
python write_files.py
```

### 4. Run the data pipeline

```powershell
python -m src.fetch_jobs          # Fetch jobs from Remotive API
python -m src.clean_jobs          # Clean and normalize job text
python -m src.embed_jobs          # Embed jobs as vectors (first run downloads ~90MB model)
python -m src.build_faiss_index   # Build FAISS index
```

### 5. Launch the app

```powershell
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), upload your resume, and click **Find My Best Jobs**.

---

## ⚙️ Configuration Reference

All settings in `config/app_config.json`. No model names, limits, or prompts are hardcoded in Python.

| Key | Default | Description |
|-----|---------|-------------|
| `models.embed_model` | `local:sentence-transformers/all-MiniLM-L6-v2` | Embedding model. Prefix `local:` = runs via sentence-transformers. Otherwise calls Together.ai API. |
| `models.chat_model` | `meta-llama/Llama-3.2-3B-Instruct-Turbo` | LLM used for fit scoring and explanation |
| `models.rerank_model` | `Salesforce/Llama-Rank-V1` | Reserved for future reranking |
| `limits.num_jobs_fetch` | `300` | Max jobs to pull from Remotive per run |
| `limits.top_k_retrieve` | `10` | FAISS nearest-neighbour results to retrieve |
| `limits.top_n_score` | `5` | Number of jobs to score with the LLM |
| `limits.llm_max_tokens` | `220` | Max tokens in each LLM scoring response |
| `limits.embed_batch_size` | `64` | Jobs to embed per API/local batch |
| `prompts.score_job.file` | `config/prompts/score_job_v2.txt` | Path to LLM prompt template |
| `prompts.score_job.output_fields` | `{fit_score: FIT_SCORE, ...}` | Maps Python dict keys → LLM output keys |
| `roles` | ML Engineer, LLM Engineer, ... | Role filter keywords shown in sidebar |
| `noise_patterns` | `[...]` | Regex list to strip boilerplate from job descriptions |
| `job_api.url` | `https://remotive.com/api/remote-jobs` | Job source API endpoint |

### Switching the embedding model

The sidebar has a built-in **"Find Embedding Models"** button that queries the Together.ai API live and lets you switch models with one click. Or edit manually:

```json
{
  "models": {
    "embed_model": "local:sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

After switching, rebuild the index:

```powershell
python -m src.embed_jobs
python -m src.build_faiss_index
```

---

## 💰 Cost Estimate

| Component | Cost | Notes |
|-----------|------|-------|
| Job embeddings | **Free** | sentence-transformers runs locally on CPU |
| Resume embedding | **Free** | Same local model, single vector |
| LLM scoring (5 jobs) | **~$0.006–0.012** | Llama-3.2-3B @ $0.06/M tokens × ~200 tokens × 5 jobs |
| Job fetching | **Free** | Remotive public API, no key required |
| **Total per run** | **~$0.01–0.02** | Mainly Together.ai LLM calls |

---

## 🗑 Files to Remove

These debug and patch scripts were created during development. They are not needed in production.

| File | Why it exists | Safe to delete? |
|------|--------------|-----------------|
| `debug_live.py` | Traced the full match → score pipeline step by step | ✅ Yes |
| `debug_match.py` | Tested `load_faiss_index` and `match_resume_to_jobs` | ✅ Yes |
| `debug_score.py` | Showed raw LLM output for parser debugging | ✅ Yes |
| `patch_app.py` | Fixed the field key mapping bug (now in `app.py`) | ✅ Yes |
| `src/extract_resume.py` | Old standalone script — now inside `match_jobs.py` | ✅ Yes |

Delete all at once:

```powershell
Remove-Item debug_live.py, debug_match.py, debug_score.py, patch_app.py -ErrorAction SilentlyContinue
```

---

## 🔮 Future Improvements

- CSV / JSON export of results
- Skill gap radar chart
- Location and salary filters
- Docker deployment
- Streamlit Cloud deployment guide
- Login system with history
- Reranking with `Salesforce/Llama-Rank-V1`

---

## 📄 License

MIT — free to use, modify, and distribute.
