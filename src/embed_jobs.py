"""
embed_jobs.py - Embeds all cleaned job descriptions using local or API embeddings.
Config: models.embed_model, limits.embed_batch_size, limits.max_resume_chars_embed
Usage: python src/embed_jobs.py
"""
import os, json
from pathlib import Path
import numpy as np
import faiss
from src.config import get_models, get_limits
from src.embedder import embed_texts

CLEAN_PATH = Path("data/jobs/jobs_clean.json")
OUT_DIR    = Path("data/index")
VEC_PATH   = OUT_DIR / "job_vectors.npy"
META_PATH  = OUT_DIR / "job_meta.json"

def main():
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Missing {CLEAN_PATH}. Run src/clean_jobs.py first.")

    models      = get_models()
    limits      = get_limits()
    embed_model = models["embed_model"]
    batch_size  = int(limits.get("embed_batch_size", 64))
    max_chars   = int(limits.get("max_resume_chars_embed", 1100))

    print(f"=== Embed Jobs ===")
    print(f"Model         : {embed_model}")
    print(f"Batch size    : {batch_size}")
    print(f"Max chars/job : {max_chars}")

    jobs  = json.loads(CLEAN_PATH.read_text(encoding="utf-8"))
    texts = [(j.get("clean_text","") or "").strip()[:max_chars] for j in jobs]
    print(f"Total jobs    : {len(texts)}")

    # Together client only needed for API models
    client = None
    if not embed_model.startswith("local:"):
        import os
        from together import Together
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise EnvironmentError("TOGETHER_API_KEY is not set.")
        client = Together(api_key=api_key)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_vecs = []
    print(f"\nEmbedding in batches of {batch_size}...")
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vecs  = embed_texts(batch, embed_model, client)
        all_vecs.append(vecs)
        print(f"  Embedded {start + len(batch):>5} / {len(texts)}")

    vecs = np.vstack(all_vecs)
    faiss.normalize_L2(vecs)
    np.save(VEC_PATH, vecs)

    meta = [{"id": j.get("id",""), "title": j.get("title",""), "company": j.get("company",""),
             "location": j.get("location",""), "url": j.get("url",""), "tags": j.get("tags",[])}
            for j in jobs]
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDONE")
    print(f"Vectors saved : {VEC_PATH}  shape={vecs.shape}")
    print(f"Meta saved    : {META_PATH}")

if __name__ == "__main__":
    main()
