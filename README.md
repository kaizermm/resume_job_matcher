# 📄 Resume Job Matcher AI

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Resume Job Matcher AI** is an AI-powered web application that analyzes a user’s resume (PDF) and automatically finds the best-matching job postings using **semantic search, embeddings, and LLM-based scoring**.

It returns **ranked job matches**, **fit scores**, and **human-readable explanations** — all inside a clean **Streamlit web app**.

---

## 📚 Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Environment Setup](#-environment-setup-togetherai)
- [Run Locally](#-run-locally)
- [Using Your Resume](#-using-your-resume)
- [Example Output](#-example-output)
- [Deploy Online](#-deploy-online-streamlit-cloud)
- [Outlier.ai Integration](#-outlierai-integration-optional)
- [Customization](#-customization-guide)
- [Cost Control](#-cost-control)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## ✨ Features

- 📄 Upload resume (PDF)
- 🔍 Semantic job matching with FAISS
- 🧠 AI-generated fit score (0–100)
- 📝 Explanation for each job match
- ⚡ Fast vector similarity search
- 💸 Runs under a **$5 budget**
- 🌐 Web UI built with Streamlit

---

## 🧠 How It Works

```text
Resume PDF
   ↓
Text Extraction
   ↓
Embeddings (Together.ai)
   ↓
FAISS Vector Search
   ↓
Top Job Matches
   ↓
LLM Scoring + Explanation
   ↓
Streamlit Web App

```


## 🗂 Project Structure
```text
resume_job_matcher/
│
├── app.py                  # Streamlit web app
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── fetch_jobs.py           # Fetch jobs (Remotive API)
│   ├── clean_jobs.py           # Clean job descriptions
│   ├── embed_jobs.py           # Embed jobs (Together.ai)
│   ├── build_faiss_index.py    # Build FAISS index
│   ├── match_jobs.py           # Resume → job matching logic
│   └── score_explain.py        # LLM scoring & explanation
│
└── data/                   # Generated at runtime (gitignored)
    ├── jobs/
    ├── index/
    └── resume/

```

* * *

## 🔑 Requirements

-   Python 3.10+
    
-   Together.ai API key
    
-   Internet connection

   

Install dependencies:

`pip install streamlit pypdf together faiss-cpu numpy`

* * *

## 🔐 Environment Setup (Together.ai)

Set your API key:

### Windows (PowerShell)

`setx TOGETHER_API_KEY "your_api_key_here"`

Restart the terminal after setting it.

Verify:

`python -c "import os; print(os.environ.get('TOGETHER_API_KEY'))"`

* * *

## 🧠 How It Works (Pipeline)

1.  Fetch job postings (Remotive API)
    
2.  Clean job descriptions
    
3.  Extract resume text from PDF
    
4.  Embed jobs and resume (Together.ai embeddings)
    
5.  Build FAISS vector index
    
6.  Retrieve top matching jobs
    
7.  Score matches using LLM
    
8.  Display results in Streamlit web app
    

* * *

## 🖥 How to Deploy Locally (Step-by-Step)

### 1\. Activate virtual environment

`.\.venv\Scripts\Activate.ps1`

### 2\. Run the Streamlit app

`streamlit run app.py`

### 3\. Open in browser

`http://localhost:8501`

* * *

## 📄 How to Use Your Own Resume

### Option A: Upload via Web App (recommended)

Upload your resume PDF directly through the Streamlit interface.

* * *

### Option B: Replace resume file manually

Put your resume PDF into:

`data/resume/`

Example:

`data/resume/My_Resume.pdf`

Then update this file:

`src/03_extract_resume.py`

Change:

`RESUME_PATH = Path("data/resume/your_resume.pdf")`

* * *

## 📊 Example Result Output

`#1 score=0.69 Senior Data Engineer — Company A Location: USA URL: https://...  Summary: Strong match in Python, SQL, and data pipelines.  #2 score=0.66 AI Engineer — Company B Location: Worldwide URL: https://...  Summary: Matches ML and AI experience but missing cloud tools.`

Displayed in the web app with:

-   Job title
    
-   Company
    
-   Location
    
-   URL
    
-   Score
    
-   Explanation
    

* * *

## 🌐 Deploy Online (Streamlit Cloud)

1.  Push project to GitHub
    
2.  Go to: [https://share.streamlit.io](https://share.streamlit.io)
    
3.  Select repository and `app.py`
    
4.  Add secret:
    

`TOGETHER_API_KEY = your_api_key_here`

5.  Deploy
    

* * *

## 🔗 Connecting to Outlier.ai (Optional Integration)

You can integrate this project with **Outlier.ai** for:

-   Data labeling
    
-   Resume-job matching evaluation
    
-   Model feedback and QA
    

### How to connect:

1.  Export results as JSON or CSV from `07_score_explain.py`
    
2.  Upload results to Outlier.ai as a dataset
    
3.  Use Outlier tasks to:
    
    -   Validate match quality
        
    -   Improve scoring rules
        
    -   Collect human feedback
        

Suggested export file:

`data/cache/scored_matches.json`

Outlier.ai can be used to:

-   Label good vs bad matches
    
-   Improve prompts
    
-   Benchmark performance
    

* * *

## ⚙️ Files to Modify for Customization

| Purpose | File |
| --- | --- |
| Web UI | `app.py` |
| Resume extraction | `src/extract_resume.py` |
| Job cleaning | `src/clean_jobs.py` |
| Embedding model | `src/embed_jobs.py` |
| Matching logic | `src/match_jobs.py` |
| Scoring & explanation | `src/score_explain.py` |

* * *

## 💰 Cost Control (Under $5 Budget)

-   Embeddings model: `m2-bert-80M-8k-retrieval`
    
-   Chat model: `Llama-3.2-3B`
    
-   Limit text length:
    

`resume_text = resume_text[:2000] job_text = job_text[:1500]`

-   Top 5 matches only
    

Estimated cost: **~$0.01 per run**

* * *

## 🛠 Future Improvements

-   Login system
    
-   CSV export
    
-   Skill gap visualization
    
-   Filters by location or role
    
-   UI charts
    
-   Reranking with LLM
    
-   Docker deployment