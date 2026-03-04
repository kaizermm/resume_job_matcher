"""
embedder.py - Unified embedding interface.
  - If embed_model starts with "local:" uses sentence-transformers (free, offline).
  - Otherwise calls Together.ai embeddings API.

Change the model in config/app_config.json at any time.
"""
import numpy as np

_local_model_cache = {}

def _get_local_model(model_name: str):
    if model_name not in _local_model_cache:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading local model: {model_name} (first run downloads ~90 MB)...")
        _local_model_cache[model_name] = SentenceTransformer(model_name)
    return _local_model_cache[model_name]


def embed_texts(texts: list, model_string: str, client=None) -> np.ndarray:
    """
    Embed a list of strings. Returns float32 numpy array of shape (N, dim).
    model_string: "local:sentence-transformers/all-MiniLM-L6-v2"  -> local
                  "BAAI/bge-base-en-v1.5"                          -> Together API
    """
    if model_string.startswith("local:"):
        model_name = model_string[len("local:"):]
        model = _get_local_model(model_name)
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype="float32")
    else:
        if client is None:
            raise ValueError("Together client required for API embedding.")
        resp = client.embeddings.create(model=model_string, input=texts)
        return np.array([d.embedding for d in resp.data], dtype="float32")


def embed_one(text: str, model_string: str, client=None) -> np.ndarray:
    """Embed a single string. Returns 1-D float32 array."""
    vecs = embed_texts([text], model_string, client)
    return vecs[0]
