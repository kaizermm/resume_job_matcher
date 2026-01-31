import streamlit as st
import tempfile
from pathlib import Path
import json

from src.extract_resume import extract_text
from src.match_jobs import load_faiss_index, match_resume_to_jobs
from src.score_explain import score_job

from together import Together
import os
from pathlib import Path
import json
from pathlib import Path

CLEAN_JOBS_PATH = Path("data/jobs/jobs_clean.json")
clean_jobs = json.loads(CLEAN_JOBS_PATH.read_text(encoding="utf-8"))



# --------------------
# Config
# --------------------
INDEX_PATH = "data/index/faiss.index"
META_PATH = "data/index/job_meta.json"

client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

st.set_page_config(page_title="Resume Job Matcher", layout="wide")
st.title("📄 Resume → Job Matcher AI")

st.write("Upload your resume and get top matching jobs with AI score & explanation.")

# --------------------
# Upload Resume
# --------------------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        resume_path = tmp.name

    st.success("Resume uploaded successfully!")

    resume_text = extract_text(Path(resume_path))

    st.subheader("Extracted Resume Text (preview)")
    st.text(resume_text[:500])

    if st.button("🔍 Find Best Jobs"):
        st.info("Matching jobs...")

        index, meta = load_faiss_index(INDEX_PATH, META_PATH)

        top_jobs = match_resume_to_jobs(resume_text, index, meta, client ,top_k=5)

        results = []

        for job in top_jobs:
            job_clean_text = clean_jobs[job["idx"]]["clean_text"]
            scoring = score_job(client, resume_text, job_clean_text)

            st.markdown(f"### {job['title']} — {job['company']}")
            st.write(f"Score: {scoring['fit_score']}/100")
            st.write(f"Matched Skills: {', '.join(scoring['matched_skills'])}")
            st.write(f"Missing Skills: {', '.join(scoring['missing_skills'])}")
            st.write(scoring["one_sentence_summary"])
            st.markdown(f"[Job link]({job['url']})")
