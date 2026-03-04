"""
match_jobs.py - Embeds the resume and retrieves top-K matching jobs via FAISS.
Config: models.embed_model, limits.top_k_retrieve, limits.max_resume_chars_embed, roles
Usage: python src/match_jobs.py
"""
import os, json
from pathlib import Path
import numpy as np
import faiss
from pypdf import PdfReader
from src.config import get_models, get_limits, get_roles
from src.embedder import embed_one

INDEX_PATH = Path("data/index/faiss.index")
META_PATH  = Path("data/index/job_meta.json")
CLEAN_PATH = Path("data/jobs/jobs_clean.json")
RESUME_DIR = Path("data/resume")

def load_faiss_index(index_path=INDEX_PATH, meta_path=META_PATH, clean_path=CLEAN_PATH):
    for p in [index_path, meta_path, clean_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing {p}. Run the full pipeline first.")
    index      = faiss.read_index(str(index_path))
    meta       = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    clean_jobs = json.loads(Path(clean_path).read_text(encoding="utf-8"))
    return index, meta, clean_jobs

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join((p.extract_text() or "") for p in reader.pages)

def load_resume_text(resume_path=None):
    if resume_path:
        p = Path(resume_path)
        if not p.exists():
            raise FileNotFoundError(f"Resume not found: {p}")
        return extract_text_from_pdf(str(p)) if p.suffix.lower() == ".pdf" else p.read_text(encoding="utf-8")
    txts = list(RESUME_DIR.glob("*.txt"))
    if txts:
        return txts[0].read_text(encoding="utf-8")
    pdfs = list(RESUME_DIR.glob("*.pdf"))
    if pdfs:
        return extract_text_from_pdf(str(pdfs[0]))
    raise FileNotFoundError(f"No resume found in {RESUME_DIR}/.")

def role_match(job, preferred_role, roles_cfg):
    if preferred_role == "Any":
        return True
    keywords = roles_cfg.get(preferred_role, [])
    haystack = " ".join([job.get("title",""), job.get("company",""),
                         job.get("location",""), " ".join(job.get("tags",[]) or [])]).lower()
    return any(k.lower() in haystack for k in keywords)

def match_resume_to_jobs(resume_text, index, meta, clean_jobs, client=None,
                          preferred_role="Any", top_k=None):
    limits    = get_limits()
    models    = get_models()
    roles_cfg = get_roles()
    if top_k is None:
        top_k = int(limits.get("top_k_retrieve", 10))
    embed_model = models["embed_model"]
    max_chars   = int(limits.get("max_resume_chars_embed", 1100))

    trimmed = resume_text.strip()[:max_chars]
    vec     = embed_one(trimmed, embed_model, client).reshape(1, -1)
    faiss.normalize_L2(vec)

    search_k = min(top_k * 3, index.ntotal)
    scores, indices = index.search(vec, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        idx = int(idx)
        if idx < 0 or idx >= len(meta):
            continue
        m   = meta[idx]
        job = clean_jobs[idx] if idx < len(clean_jobs) else {}
        if not role_match(m, preferred_role, roles_cfg):
            continue
        results.append({"rank": len(results)+1, "score": float(score), "idx": idx,
                        "title": m.get("title",""), "company": m.get("company",""),
                        "location": m.get("location",""), "url": m.get("url",""),
                        "tags": m.get("tags",[]), "clean_text": job.get("clean_text","")})
        if len(results) >= top_k:
            break

    if len(results) < 3 and preferred_role != "Any":
        results = []
        for score, idx in zip(scores[0], indices[0]):
            idx = int(idx)
            if idx < 0 or idx >= len(meta):
                continue
            m   = meta[idx]
            job = clean_jobs[idx] if idx < len(clean_jobs) else {}
            results.append({"rank": len(results)+1, "score": float(score), "idx": idx,
                            "title": m.get("title",""), "company": m.get("company",""),
                            "location": m.get("location",""), "url": m.get("url",""),
                            "tags": m.get("tags",[]), "clean_text": job.get("clean_text","")})
            if len(results) >= top_k:
                break
    return results

if __name__ == "__main__":
    print("Use: python src/match_jobs.py --role \"ML Engineer\"")
