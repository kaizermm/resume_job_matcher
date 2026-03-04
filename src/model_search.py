"""
model_search.py - Queries Together.ai API to find available serverless embedding models.
Also supports switching the active embed_model in config/app_config.json.
"""
import os
import json
import requests
from pathlib import Path

TOGETHER_MODELS_URL = "https://api.together.ai/v1/models"
CONFIG_PATH = Path("config/app_config.json")


def fetch_together_models(api_key: str) -> list:
    """Call Together.ai /v1/models and return the full model list."""
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(TOGETHER_MODELS_URL, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


def filter_embedding_models(models: list) -> list:
    """
    Filter to embedding-type serverless models only.
    Returns list of dicts with keys: id, display_name, context_length, pricing.
    """
    results = []
    for m in models:
        # Together API uses "type" or "model_type" field
        model_type = (m.get("type") or m.get("model_type") or "").lower()
        if "embed" not in model_type:
            continue

        # Skip non-serverless (dedicated only)
        pricing = m.get("pricing") or {}
        input_price = pricing.get("input", None)

        results.append({
            "id":             m.get("id", ""),
            "display_name":   m.get("display_name") or m.get("name") or m.get("id", ""),
            "context_length": m.get("context_length", "?"),
            "input_price":    input_price,
            "pricing_str":    f"${input_price}/M tokens" if input_price else "free/unknown",
        })

    results.sort(key=lambda x: x["id"])
    return results


def get_available_embedding_models(api_key: str) -> tuple:
    """
    Returns (models_list, error_string).
    models_list is [] on error. error_string is None on success.
    """
    try:
        all_models = fetch_together_models(api_key)
        embed_models = filter_embedding_models(all_models)
        return embed_models, None
    except requests.exceptions.HTTPError as e:
        return [], f"API error {e.response.status_code}: {e.response.text[:200]}"
    except requests.exceptions.ConnectionError:
        return [], "Connection error — check your internet connection."
    except requests.exceptions.Timeout:
        return [], "Request timed out."
    except Exception as e:
        return [], str(e)


def set_embed_model(model_id: str) -> str:
    """
    Update embed_model in config/app_config.json.
    Returns the new model id on success, raises on failure.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["models"]["embed_model"] = model_id
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return model_id


def get_current_embed_model() -> str:
    """Return the current embed_model from config."""
    if not CONFIG_PATH.exists():
        return "unknown"
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return cfg.get("models", {}).get("embed_model", "unknown")
