import json
import requests
from pathlib import Path

OUT = Path("data/jobs")
OUT.mkdir(parents=True, exist_ok=True)

URL = "https://remotive.com/api/remote-jobs"

def main():
    r = requests.get(URL, timeout=30)
    r.raise_for_status()
    jobs = r.json().get("jobs", [])
    out_path = OUT / "jobs_raw.json"
    out_path.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(jobs)} jobs to {out_path}")

if __name__ == "__main__":
    main()
