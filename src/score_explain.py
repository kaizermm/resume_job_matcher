import os, re
from pathlib import Path

OUT_DIR  = Path("data/cache")
OUT_PATH = OUT_DIR / "scored_matches.json"

def trim(text, max_chars):
    return (text or "").strip()[:max_chars]

def grab(lines, key):
    key_upper = key.strip().upper()
    for line in lines:
        clean = re.sub(r"[*`]+", "", line).strip()
        cu = clean.upper()
        if cu.startswith(key_upper + ":"):
            return clean[len(key_upper)+1:].strip()
        m = re.match(rf"^{re.escape(key_upper)}\s*[-]\s*(.+)$", cu)
        if m:
            return clean[m.start(1):].strip()
    return ""

def parse_kv_output(raw, fields):
    lines = [l for l in raw.strip().splitlines() if l.strip()]
    def to_list(val):
        if not val or val.strip().upper() in ("N/A", "NONE", "-", ""):
            return []
        return [p.strip(" -*\t") for p in re.split(r"[,;]+", val)
                if p.strip(" -*\t")]
    result = {}
    for python_key, config_key in fields.items():
        raw_val = grab(lines, config_key)
        if python_key == "fit_score":
            m = re.search(r"\b(\d{1,3})\b", raw_val)
            result[python_key] = max(0, min(100, int(m.group(1)) if m else 0))
        elif python_key in ("matched_skills", "missing_skills", "recommendations"):
            result[python_key] = to_list(raw_val)
        elif python_key == "summary":
            result[python_key] = raw_val.replace("\n", " ").strip() or "No summary."
        else:
            result[python_key] = raw_val
    return result

def score_job(resume_text, job_text, client, prompt_template, fields,
              model, max_tokens, max_resume_chars, max_job_chars, debug=False):
    prompt = prompt_template.format(
        resume_text=trim(resume_text, max_resume_chars),
        job_text=trim(job_text, max_job_chars),
    )
    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=max_tokens,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        empty = {k: (0 if k == "fit_score" else
                    ([] if k in ("matched_skills","missing_skills","recommendations") else ""))
                 for k in fields}
        empty["summary"] = f"LLM error: {e}"
        return empty
    if debug:
        print(f"  RAW: {repr(raw[:300])}")
    result = parse_kv_output(raw, fields)
    result["raw_llm_output"] = raw
    return result

def score_top_jobs(resume_text, matches, client, top_n=None, debug=False):
    from src.config import get_models, get_limits, load_prompt, get_prompt_fields
    limits = get_limits()
    models = get_models()
    if top_n is None:
        top_n = int(limits.get("top_n_score", 5))
    model            = models["chat_model"]
    max_tokens       = int(limits.get("llm_max_tokens", 220))
    max_resume_chars = int(limits.get("max_resume_chars_prompt", 1500))
    max_job_chars    = int(limits.get("max_job_chars_prompt", 1500))
    prompt_template  = load_prompt("score_job")
    fields           = get_prompt_fields("score_job")
    scored = []
    for job in matches[:top_n]:
        title   = job.get("title", "?")
        company = job.get("company", "?")
        print(f"  Scoring: {title} @ {company} ...")
        result = score_job(
            resume_text, job.get("clean_text", ""), client,
            prompt_template, fields, model, max_tokens,
            max_resume_chars, max_job_chars, debug=debug,
        )
        if debug:
            print(f"    fit_score={result.get(chr(102)+chr(105)+chr(116)+chr(95)+chr(115)+chr(99)+chr(111)+chr(114)+chr(101))}")
        enriched = {**job, **result, "rank": len(scored) + 1}
        scored.append(enriched)
    return scored
