import json
from pathlib import Path

CONFIG_PATH = Path("config/app_config.json")
_cache = {}

def load_config():
    if "config" not in _cache:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Missing config file: {CONFIG_PATH.resolve()}")
        _cache["config"] = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return _cache["config"]

def load_prompt(prompt_key):
    cfg = load_config()
    prompt_cfg = cfg.get("prompts", {}).get(prompt_key)
    if not prompt_cfg:
        raise KeyError(f"Prompt key '{prompt_key}' not found in config['prompts'].")
    prompt_file = Path(prompt_cfg["file"])
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file.resolve()}")
    return prompt_file.read_text(encoding="utf-8").strip()

def get_prompt_fields(prompt_key):
    cfg = load_config()
    return cfg.get("prompts", {}).get(prompt_key, {}).get("output_fields", {})

def get_prompt_version(prompt_key):
    cfg = load_config()
    return cfg.get("prompts", {}).get(prompt_key, {}).get("file", f"config/prompts/{prompt_key}.txt")

def get_roles():
    return load_config().get("roles", {})

def get_role_names():
    return list(get_roles().keys())

def get_limits():
    return load_config().get("limits", {})

def get_models():
    return load_config().get("models", {})

def get_noise_patterns():
    return load_config().get("noise_patterns", [])

def get_job_api_config():
    return load_config().get("job_api", {})
