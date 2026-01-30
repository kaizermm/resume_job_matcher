from pathlib import Path
import numpy as np
import faiss

VEC_PATH = Path("data/index/job_vectors.npy")
INDEX_PATH = Path("data/index/faiss.index")

def main():
    if not VEC_PATH.exists():
        raise FileNotFoundError(f"Missing {VEC_PATH}. Run Step 7 first.")

    vecs = np.load(VEC_PATH).astype("float32")

    n, dim = vecs.shape
    print(f"Loaded vectors: {VEC_PATH}  shape=({n}, {dim})")

    # Using inner product (works like cosine similarity because we normalized in Step 7)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    print("DONE ✅")
    print(f"Saved FAISS index: {INDEX_PATH}")
    print(f"Total vectors in index: {index.ntotal}")

if __name__ == "__main__":
    main()
