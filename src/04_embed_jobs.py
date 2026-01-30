import os
import json
import time
from pathlib import Path

import numpy as np
import faiss
from together import Together

# ----- Paths -----
CLEAN_PATH = Path("data/jobs/jobs_clean.json")
OUT_DIR = Path("data/index")
VEC_PATH = OUT_DIR / "job_vectors.npy"
META_PATH = OUT_DIR / "job_meta.json"

# ----- Together model -----
EMBED_MODEL = "BAAI/bge-base-en-v1.5"  # good cheap embedding model
BATCH_SIZE = 64                        # safe batch size

def trim_for_embedding(text: str, max_chars: int = 1400) -> str:
    # 1400 chars is usually safely under 512 tokens for English text
    text = (text or "").strip()
    return text[:max_chars]


def embed_batch(client: Together, texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    # resp.data is a list of objects with `.embedding`
    return np.array([d.embedding for d in resp.data], dtype="float32")

def main():
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise EnvironmentError("TOGETHER_API_KEY is not set in this terminal session.")

    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Missing {CLEAN_PATH}. Run Step 5 first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jobs = json.loads(CLEAN_PATH.read_text(encoding="utf-8"))
    texts = [trim_for_embedding(j.get("clean_text", ""), max_chars=1400) for j in jobs]


    # Basic validation
    empty_count = sum(1 for t in texts if not t.strip())
    if empty_count > 0:
        print(f"Warning: {empty_count} jobs have empty clean_text. They will embed poorly.")

    client = Together(api_key=api_key)

    all_vecs = []
    print(f"Embedding {len(texts)} jobs in batches of {BATCH_SIZE}...")

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        # Simple retry (helps with occasional network hiccups)
        for attempt in range(3):
            try:
                vecs = embed_batch(client, batch)
                all_vecs.append(vecs)
                print(f"  Embedded {start + len(batch):>5}/{len(texts)}")
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  Retry batch {start}-{start+len(batch)} بسبب: {e}")
                time.sleep(2 * (attempt + 1))

    vecs = np.vstack(all_vecs)

    # Normalize vectors so FAISS inner product behaves like cosine similarity
    faiss.normalize_L2(vecs)

    # Save vectors
    np.save(VEC_PATH, vecs)

    # Save metadata (so you can map vector index -> job info)
    meta = [
        {
            "id": j.get("id", ""),
            "title": j.get("title", ""),
            "company": j.get("company", ""),
            "location": j.get("location", ""),
            "url": j.get("url", ""),
            "tags": j.get("tags", []),
        }
        for j in jobs
    ]
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDONE ✅")
    print(f"Saved vectors: {VEC_PATH}  shape={vecs.shape}")
    print(f"Saved meta:    {META_PATH}")

if __name__ == "__main__":
    main()
