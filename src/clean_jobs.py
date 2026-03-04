import json, re
from pathlib import Path
from src.config import get_limits, get_noise_patterns

RAW_PATH   = Path("data/jobs/jobs_raw.json")
OUT_DIR    = Path("data/jobs")
CLEAN_PATH = OUT_DIR / "jobs_clean.json"

def strip_html(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    return text.replace("&nbsp;"," ").replace("&amp;","&").replace("&lt;","<").replace("&gt;",">")

def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text or "").strip()

def remove_noise(text, patterns):
    out = text
    for pattern in patterns:
        out = re.sub(pattern, " ", out, flags=re.IGNORECASE)
    return normalize_whitespace(out)

def build_clean_text(title, company, location, tags, description):
    tags_part = ", ".join(tags) if tags else ""
    parts = [f"Title: {title}", f"Company: {company}",
             f"Location: {location}" if location else "",
             f"Tags: {tags_part}" if tags_part else "",
             "Description:", description]
    return "\n".join(p for p in parts if p)

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing {RAW_PATH}. Run src/fetch_jobs.py first.")
    limits         = get_limits()
    noise_patterns = get_noise_patterns()
    max_chars      = int(limits.get("max_job_chars_clean", 2500))
    raw_data = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    raw      = raw_data.get("jobs", raw_data) if isinstance(raw_data, dict) else raw_data
    print(f"Input  : {len(raw)} raw jobs")
    clean = []
    for j in raw:
        title    = j.get("title","").strip()
        company  = j.get("company_name","").strip()
        location = j.get("candidate_required_location","").strip()
        url      = j.get("url","")
        tags     = j.get("tags",[]) or []
        desc     = strip_html(j.get("description","") or "")
        desc     = remove_noise(desc, noise_patterns)[:max_chars]
        clean.append({"id": str(j.get("id","")), "title": title, "company": company,
                      "location": location, "url": url, "tags": tags,
                      "clean_text": build_clean_text(title, company, location, tags, desc)})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_PATH.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Output : {len(clean)} clean jobs -> {CLEAN_PATH}")

if __name__ == "__main__":
    main()
