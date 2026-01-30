import os
import json
from pathlib import Path

import numpy as np
import faiss
from together import Together
from pypdf import PdfReader

# ----- Paths -----
INDEX_PATH = Path("data/index/faiss.index")
META_PATH = Path("data/index/job_meta.json")
RESUME_TXT = Path("data/resume/resume.txt")
RESUME_PDF = Path("data/resume/resume.pdf")

# ----- Together model -----
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

def trim_for_embedding(text: str, max_chars: int = 1100) -> str:
    # Keep safely under 512 tokens for embeddings
    text = (text or "").strip()
    return text[:max_chars]

def read_resume_text() -> str:
    # Prefer resume.txt if it exists (best quality)
    if RESUME_TXT.exists():
        return RESUME_TXT.read_text(encoding="utf-8")

    # Fallback: extract from PDF if resume.txt doesn't exist
    if RESUME_PDF.exists():
        reader = PdfReader(str(RESUME_PDF))
        pages = [(p.extract_text() or "") for p in reader.pages]
        return "\n".join(pages)

    raise FileNotFoundError(
        "Could not find resume.txt or resume.pdf. Put your resume at:\n"
        "- data/resume/resume.txt OR data/resume/resume.pdf"
    )

def embed_text(client: Together, text: str) -> np.ndarray:
    text = trim_for_embedding(text)
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    # Normalize so inner product = cosine similarity
    vec /= (np.linalg.norm(vec) + 1e-12)
    return vec

def main(top_k: int = 10):
    # Check API key
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise EnvironmentError("TOGETHER_API_KEY is not set in this terminal session.")

    # Check files exist
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing {INDEX_PATH}. Run Step 8 first.")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing {META_PATH}. Run Step 7 first.")

    # Load index + metadata
    index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    # Read + embed resume
    resume_text = read_resume_text()
    resume_text = " ".join(resume_text.split())  # light cleanup
    client = Together(api_key=api_key)
    q = embed_text(client, resume_text).reshape(1, -1)

    # Search
    top_k = min(top_k, index.ntotal)
    scores, idxs = index.search(q, top_k)

    print("\nTop matches ✅")
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        j = meta[int(idx)]
        title = j.get("title", "")
        company = j.get("company", "")
        location = j.get("location", "")
        url = j.get("url", "")
        print(f"\n#{rank}  score={float(score):.4f}")
        print(f"{title} — {company}")
        if location:
            print(f"Location: {location}")
        if url:
            print(f"URL: {url}")

if __name__ == "__main__":
    main(top_k=10)
