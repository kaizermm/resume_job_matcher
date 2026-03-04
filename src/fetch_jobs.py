import json, time, argparse
from pathlib import Path
import requests
from src.config import get_limits, get_job_api_config, get_roles

OUT_DIR = Path("data/jobs")

def fetch_raw_jobs(api_cfg):
    url         = api_cfg["url"]
    timeout     = api_cfg.get("timeout_seconds", 30)
    max_retries = api_cfg.get("max_retries", 3)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Fetching from {url} (attempt {attempt})...")
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json().get("jobs", [])
        except Exception as e:
            last_err = e
            wait = 1.5 * attempt
            print(f"  Error: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} tries: {last_err}")

def job_matches_role(job, role, roles_cfg):
    if role == "Any":
        return True
    keywords = roles_cfg.get(role, [])
    if not keywords:
        return True
    haystack = " ".join([str(job.get("title","")), str(job.get("company_name","")),
                         " ".join(job.get("tags",[]) or [])]).lower()
    return any(k.lower() in haystack for k in keywords)

def main(preferred_role="Any"):
    limits    = get_limits()
    api_cfg   = get_job_api_config()
    roles_cfg = get_roles()
    num_jobs  = int(limits["num_jobs_fetch"])
    print(f"Target: {num_jobs} jobs  |  Role: {preferred_role}")
    all_jobs = fetch_raw_jobs(api_cfg)
    print(f"Fetched: {len(all_jobs)}")
    filtered = [j for j in all_jobs if job_matches_role(j, preferred_role, roles_cfg)]
    if preferred_role != "Any" and len(filtered) < 25:
        print(f"  Filter too strict ({len(filtered)}). Using all jobs.")
        filtered = all_jobs
    limited = filtered[:num_jobs]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "jobs_raw.json"
    out_path.write_text(json.dumps(limited, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(limited)} jobs -> {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default="Any")
    args = parser.parse_args()
    main(preferred_role=args.role)
