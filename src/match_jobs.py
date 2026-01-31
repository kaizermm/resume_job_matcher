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

EMBED_MODEL = "BAAI/bge-base-en-v1.5"


def trim_for_embedding(text: str, max_chars: int = 1100) -> str:
    return (text or "").strip()[:max_chars]


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n".join(pages)


def embed_text(client: Together, text: str) -> np.ndarray:
    text = trim_for_embedding(text)
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    vec /= (np.linalg.norm(vec) + 1e-12)
    return vec


def load_faiss_index(index_path=INDEX_PATH, meta_path=META_PATH):
    if not Path(index_path).exists():
        raise FileNotFoundError(f"Missing {index_path}")
    if not Path(meta_path).exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    index = faiss.read_index(str(index_path))
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    return index, meta


def match_resume_to_jobs(resume_text: str, index, meta, client, top_k=5):
    resume_text = " ".join(resume_text.split())
    q = embed_text(client, resume_text).reshape(1, -1)

    top_k = min(top_k, index.ntotal)
    scores, idxs = index.search(q, top_k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        j = meta[int(idx)].copy()
        j["vector_score"] = float(score)
        j["idx"] = idx
        results.append(j)

    return results



def main(top_k: int = 10):
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise EnvironmentError("TOGETHER_API_KEY is not set.")

    index, meta = load_faiss_index()

    resume_text = extract_text_from_pdf("data/resume/resume.pdf")

    client = Together(api_key=api_key)
    results = match_resume_to_jobs(resume_text, index, meta, client, top_k)

    print("\nTop matches ✅")
    for i, j in enumerate(results, 1):
        print(f"\n#{i} score={j['vector_score']:.4f}")
        print(f"{j['title']} — {j['company']}")
        print(f"Location: {j['location']}")
        print(f"URL: {j['url']}")


if __name__ == "__main__":
    main(top_k=10)
