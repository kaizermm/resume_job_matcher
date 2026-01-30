import json
import re
from pathlib import Path

RAW_PATH = Path("data/jobs/jobs_raw.json")
OUT_DIR = Path("data/jobs")
CLEAN_PATH = OUT_DIR / "jobs_clean.json"

def strip_html(text: str) -> str:
    # remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text or "")
    # unescape common html entities (basic)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return text

def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text

def remove_common_noise(text: str) -> str:
    # delete common legal/EEO spam blocks (best-effort)
    patterns = [
        r"equal opportunity employer.*",
        r"we are an equal opportunity employer.*",
        r"accommodation.*disability.*",
        r"all qualified applicants.*",
        r"background check.*",
    ]
    out = text
    for p in patterns:
        out = re.sub(p, " ", out, flags=re.IGNORECASE)
    return normalize_whitespace(out)

def build_clean_text(title, company, location, tags, description):
    tags_part = ", ".join(tags) if tags else ""
    parts = [
        f"Title: {title}",
        f"Company: {company}",
        f"Location: {location}" if location else "",
        f"Tags: {tags_part}" if tags_part else "",
        "Description:",
        description
    ]
    return "\n".join([p for p in parts if p])

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing {RAW_PATH}. Run Step 4 first.")

    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))

    clean = []
    for j in raw:
        # Remotive fields
        job_id = str(j.get("id", ""))
        title = j.get("title", "").strip()
        company = j.get("company_name", "").strip()
        location = j.get("candidate_required_location", "").strip()
        url = j.get("url", "")
        tags = j.get("tags", []) or []

        desc = j.get("description", "") or ""
        desc = strip_html(desc)
        desc = remove_common_noise(desc)

        # keep description short to control cost later
        desc = desc[:2500]  # character cap (safe & simple)

        clean_text = build_clean_text(title, company, location, tags, desc)

        clean.append({
            "id": job_id,
            "title": title,
            "company": company,
            "location": location,
            "url": url,
            "tags": tags,
            "clean_text": clean_text
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_PATH.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Raw jobs:   {len(raw)}")
    print(f"Clean jobs: {len(clean)}")
    print(f"Saved to:   {CLEAN_PATH}")

if __name__ == "__main__":
    main()
