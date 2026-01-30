import streamlit as st
import numpy as np
import json
import faiss
from pypdf import PdfReader
from together import Together
import os
import re

# Load API key
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

# Load job vectors and metadata
job_vectors = np.load("data/index/job_vectors.npy")
with open("data/index/job_meta.json", "r", encoding="utf-8") as f:
    job_meta = json.load(f)

index = faiss.read_index("data/index/faiss.index")

CHAT_MODEL = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
EMBED_MODEL = "togethercomputer/m2-bert-80M-8k-retrieval"

st.title("📄 Resume Job Matcher AI")
st.write("Upload your resume PDF and get top matching jobs with scores")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text[:2000]

def embed_text(text):
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return np.array(resp.data[0].embedding, dtype="float32")

def build_prompt(resume, job):
    return f"""
Return ONLY:

FIT_SCORE: 0-100
SUMMARY: one sentence

RESUME: {resume}
JOB: {job}
"""

def parse_output(text):
    score = 0
    summary = ""
    for line in text.splitlines():
        if line.startswith("FIT_SCORE"):
            score = int(re.findall(r"\d+", line)[0])
        if line.startswith("SUMMARY"):
            summary = line.split(":",1)[1].strip()
    return score, summary

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.success("Resume uploaded successfully!")

    if st.button("🔍 Find Matching Jobs"):
        with st.spinner("Matching jobs..."):
            resume_vec = embed_text(resume_text)

            D, I = index.search(np.array([resume_vec]), 5)

            results = []
            for idx in I[0]:
                job = job_meta[idx]

                prompt = build_prompt(resume_text, job["clean_text"])
                resp = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role":"user","content":prompt}],
                    temperature=0
                )
                output = resp.choices[0].message.content
                score, summary = parse_output(output)

                results.append((score, job, summary))

            results.sort(reverse=True)

            for i, (score, job, summary) in enumerate(results,1):
                st.subheader(f"#{i} {job['title']} — {job['company']}")
                st.write(f"📍 {job['location']}")
                st.write(f"🔗 {job['url']}")
                st.write(f"✅ Score: {score}")
                st.write(f"🧠 {summary}")
                st.divider()
