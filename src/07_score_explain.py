import os
import json
from pathlib import Path

import numpy as np
import faiss
import re
from together import Together
from pypdf import PdfReader

# ---------- Paths ----------
INDEX_PATH = Path("data/index/faiss.index")
META_PATH = Path("data/index/job_meta.json")
CLEAN_JOBS_PATH = Path("data/jobs/jobs_clean.json")

RESUME_TXT = Path("data/resume/resume.txt")
RESUME_DIR = Path("data/resume")

OUT_DIR = Path("data/cache")
OUT_PATH = OUT_DIR / "scored_matches.json"

# ---------- Models ----------
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
CHAT_MODEL = "meta-llama/Llama-3.2-3B-Instruct-Turbo"  # cheap + good enough for scoring

def trim(text: str, max_chars: int) -> str:
    return (text or "").strip()[:max_chars]

def read_resume_text() -> str:
    if RESUME_TXT.exists():
        return RESUME_TXT.read_text(encoding="utf-8")

    # fallback: find any pdf in data/resume
    pdfs = list(RESUME_DIR.glob("*.pdf"))
    if pdfs:
        reader = PdfReader(str(pdfs[0]))
        pages = [(p.extract_text() or "") for p in reader.pages]
        return "\n".join(pages)

    raise FileNotFoundError("No resume.txt or resume PDF found in data/resume")

def embed_resume(client: Together, resume_text: str) -> np.ndarray:
    resume_text = trim(resume_text, 1100)  # safe under 512 token limit
    resp = client.embeddings.create(model=EMBED_MODEL, input=[resume_text])
    v = np.array(resp.data[0].embedding, dtype="float32")
    v /= (np.linalg.norm(v) + 1e-12)
    return v.reshape(1, -1)

def search_top_matches(index, q_vec, top_k: int):
    top_k = min(top_k, index.ntotal)
    scores, idxs = index.search(q_vec, top_k)
    return scores[0], idxs[0]

def build_prompt(resume_text: str, job_text: str) -> str:
    resume_text = trim(resume_text, 1500)
    job_text = trim(job_text, 1500)

    return f"""
You are scoring how well a RESUME matches a JOB.

Return EXACTLY in this format (no extra lines, no markdown):

FIT_SCORE: <0-100 integer>
MATCHED_SKILLS: <comma-separated list>
MISSING_SKILLS: <comma-separated list>
RECOMMENDATIONS: <comma-separated list>
SUMMARY: <one sentence>

RESUME: {resume_text}
JOB: {job_text}
""".strip()

def parse_kv_output(text: str) -> dict:
    # Extract lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    data = {
        "fit_score": 0,
        "matched_skills": [],
        "missing_skills": [],
        "recommendations": [],
        "one_sentence_summary": ""
    }

    def grab(prefix: str) -> str:
        for ln in lines:
            if ln.upper().startswith(prefix):
                return ln.split(":", 1)[1].strip()
        return ""

    # Fit score
    fs = grab("FIT_SCORE")
    m = re.search(r"\d+", fs)
    data["fit_score"] = int(m.group()) if m else 0
    data["fit_score"] = max(0, min(100, data["fit_score"]))

    # Lists
    def split_list(s: str):
        if not s:
            return []
        # split by commas, clean, remove empties
        items = [x.strip(" -•\t") for x in s.split(",")]
        return [x for x in items if x]

    data["matched_skills"] = split_list(grab("MATCHED_SKILLS"))
    data["missing_skills"] = split_list(grab("MISSING_SKILLS"))
    data["recommendations"] = split_list(grab("RECOMMENDATIONS"))

    # Summary
    summary = grab("SUMMARY")
    summary = summary.replace("\n", " ").strip()
    data["one_sentence_summary"] = summary

    return data

def score_job(client: Together, resume_text: str, job_text: str) -> dict:
    prompt = build_prompt(resume_text, job_text)

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=220,
    )

    content = resp.choices[0].message.content.strip()

    # Parse KV format into JSON-safe dict
    return parse_kv_output(content)


def main(top_k_retrieve: int = 10, top_n_score: int = 5):
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise EnvironmentError("TOGETHER_API_KEY is not set in this terminal session.")

    if not INDEX_PATH.exists():
        raise FileNotFoundError("Missing FAISS index. Run Step 8 first.")
    if not META_PATH.exists():
        raise FileNotFoundError("Missing job_meta.json. Run Step 7 first.")
    if not CLEAN_JOBS_PATH.exists():
        raise FileNotFoundError("Missing jobs_clean.json. Run Step 5 first.")

    index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    clean_jobs = json.loads(CLEAN_JOBS_PATH.read_text(encoding="utf-8"))

    resume_text = " ".join(read_resume_text().split())

    client = Together(api_key=api_key)
    q_vec = embed_resume(client, resume_text)

    scores, idxs = search_top_matches(index, q_vec, top_k_retrieve)

    # Score only top N to keep budget low
    top_n_score = min(top_n_score, len(idxs))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"\nScoring top {top_n_score} matches (out of {len(idxs)} retrieved)...\n")

    for rank in range(top_n_score):
        idx = int(idxs[rank])
        sim = float(scores[rank])

        j_meta = meta[idx]
        j_clean = clean_jobs[idx]["clean_text"]  # same ordering as embeddings

        print(f"#{rank+1}  sim={sim:.4f}  {j_meta.get('title','')} — {j_meta.get('company','')}")

        scoring = score_job(client, resume_text, j_clean)

        results.append({
            "rank": rank + 1,
            "similarity": sim,
            "job": j_meta,
            "scoring": scoring
        })

    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDONE ✅")
    print(f"Saved scored results to: {OUT_PATH}\n")

    # Print a small readable preview
    for r in results:
        title = r["job"].get("title", "")
        company = r["job"].get("company", "")
        fit = r["scoring"].get("fit_score", "NA")
        summary = r["scoring"].get("one_sentence_summary", "")
        print(f"- #{r['rank']} Fit={fit} | {title} — {company}")
        if summary:
            print(f"  {summary}")

if __name__ == "__main__":
    main(top_k_retrieve=10, top_n_score=5)
