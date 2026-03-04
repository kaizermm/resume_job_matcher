"""
app.py - Streamlit web app for the Resume Job Matcher.

Config-driven:
  - Role choices     : config['roles']
  - Number of jobs   : config['limits']['num_jobs_fetch']
  - Top-K / Top-N    : config['limits']['top_k_retrieve', 'top_n_score']
  - All LLM prompts  : config/prompts/*.txt
  - Output fields    : config['prompts']['score_job']['output_fields']
  - Embed model      : config['models']['embed_model']  (switchable via sidebar)
"""
import os
import json
import tempfile
from pathlib import Path

import streamlit as st
from together import Together

from src.config import get_role_names, get_limits, get_models, get_prompt_fields, get_prompt_version
from src.match_jobs import load_faiss_index, match_resume_to_jobs
from src.score_explain import score_top_jobs
from src.model_search import get_available_embedding_models, set_embed_model, get_current_embed_model

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Job Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #0f0f1a; }
  .stApp { background: #0f0f1a; color: #e0e0f0; }
  h1, h2, h3 { color: #a78bfa; }
  .metric-card {
    background: #1a1a2e;
    border: 1px solid #2d2d4e;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
  }
  .score-high  { color: #4ade80; font-size: 2rem; font-weight: 700; }
  .score-mid   { color: #facc15; font-size: 2rem; font-weight: 700; }
  .score-low   { color: #f87171; font-size: 2rem; font-weight: 700; }
  .tag-pill {
    display: inline-block;
    background: #2d2d4e;
    color: #a78bfa;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 2px;
  }
  .skill-matched { color: #4ade80; }
  .skill-missing { color: #f87171; }
  .model-pill {
    display: inline-block;
    background: #1e1e3a;
    border: 1px solid #4a4a7a;
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.75rem;
    color: #a0a0c0;
    margin: 2px 0;
    word-break: break-all;
  }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading job index...")
def get_index():
    return load_faiss_index()


@st.cache_resource(show_spinner="Connecting to Together.ai...")
def get_client():
    api_key = os.environ.get("TOGETHER_API_KEY", "")
    if not api_key:
        st.error(
            "TOGETHER_API_KEY is not set. "
            "Add it as an environment variable or Streamlit secret."
        )
        st.stop()
    return Together(api_key=api_key)


def extract_text_from_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".txt":
        return uploaded_file.read().decode("utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            reader = PdfReader(tmp_path)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            st.error("pypdf is required. Run: pip install pypdf")
            st.stop()
    st.error(f"Unsupported file type: {suffix}. Upload a .pdf or .txt file.")
    st.stop()


def score_color_class(score: int) -> str:
    if score >= 70:
        return "score-high"
    if score >= 45:
        return "score-mid"
    return "score-low"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")

    role_options   = ["Any"] + get_role_names()
    preferred_role = st.selectbox(
        "Preferred Role",
        options=role_options,
        index=0,
        help="Filter and rank jobs toward this role type.",
    )

    limits = get_limits()
    top_k = st.slider(
        "Candidates to retrieve (FAISS)",
        min_value=5, max_value=50,
        value=int(limits.get("top_k_retrieve", 10)),
        step=5,
    )
    top_n = st.slider(
        "Jobs to score with LLM",
        min_value=1, max_value=min(top_k, 15),
        value=int(limits.get("top_n_score", 5)),
        step=1,
        help="LLM scoring costs ~$0.001 per job.",
    )

    st.divider()

    # ── Model info ─────────────────────────────────────────────────────────
    current_embed = get_current_embed_model()
    st.caption(f"**Chat model:** `{get_models()['chat_model']}`")
    st.markdown(
        f'''<div class="model-pill">Embed: {current_embed}</div>''',
        unsafe_allow_html=True,
    )
    st.caption(f"Prompts: `{get_prompt_version('score_job')}`")

    st.divider()

    # ── Model search panel ─────────────────────────────────────────────────
    st.markdown("### 🔍 Find Embedding Models")
    st.caption("Search Together.ai for currently available serverless embedding models.")

    if st.button("Search available models", use_container_width=True):
        api_key = os.environ.get("TOGETHER_API_KEY", "")
        if not api_key:
            st.error("Set TOGETHER_API_KEY first.")
        else:
            with st.spinner("Querying Together.ai API..."):
                models_found, err = get_available_embedding_models(api_key)

            if err:
                st.error(f"Error: {err}")
                st.info(
                    "Together.ai may have removed all serverless embedding models. "
                    "Use local embeddings instead by keeping the current model "
                    "as `local:sentence-transformers/all-MiniLM-L6-v2`."
                )
            elif not models_found:
                st.warning(
                    "No serverless embedding models found on Together.ai. "
                    "Your best option is the free local model already configured."
                )
                st.code("local:sentence-transformers/all-MiniLM-L6-v2")
            else:
                st.success(f"Found {len(models_found)} embedding model(s):")
                st.session_state["found_models"] = models_found

    # Show found models as selectable options
    if "found_models" in st.session_state and st.session_state["found_models"]:
        found = st.session_state["found_models"]
        model_labels = [f"{m['id']} ({m['pricing_str']})" for m in found]
        selected_label = st.selectbox(
            "Select a model to activate",
            options=["— keep current —"] + model_labels,
            key="model_picker",
        )

        if selected_label != "— keep current —" and st.button(
            "Apply selected model", use_container_width=True
        ):
            chosen_id = found[model_labels.index(selected_label)]["id"]
            try:
                set_embed_model(chosen_id)
                st.success(
                    f"Model updated to `{chosen_id}`. "
                    "Re-run `python src/embed_jobs.py` and "
                    "`python src/build_faiss_index.py` to rebuild the index."
                )
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Failed to update config: {e}")

    # Always show local fallback option
    with st.expander("Use free local model (no API needed)"):
        st.markdown("**`sentence-transformers/all-MiniLM-L6-v2`**")
        st.caption("Fast, free, runs offline. ~90MB download on first use.")
        if st.button("Switch to local model", use_container_width=True):
            try:
                set_embed_model("local:sentence-transformers/all-MiniLM-L6-v2")
                st.success(
                    "Switched to local model. "
                    "Re-run embed_jobs.py and build_faiss_index.py to rebuild the index."
                )
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🎯 Resume Job Matcher")
st.markdown(
    "Upload your resume, pick a role, and get semantically-ranked job matches "
    "with AI-powered fit scores."
)

uploaded = st.file_uploader(
    "📄 Upload your resume (PDF or TXT)",
    type=["pdf", "txt"],
    help="Your file stays local — only the text is sent for embedding.",
)

run_btn = st.button("🚀 Find My Best Jobs", type="primary", disabled=uploaded is None)

if run_btn and uploaded:
    resume_text = extract_text_from_upload(uploaded)

    if len(resume_text.strip()) < 100:
        st.error("Resume text is too short — check that the PDF extracted correctly.")
        st.stop()

    client = get_client()

    try:
        index, meta, clean_jobs = get_index()
    except FileNotFoundError as e:
        st.error(
            f"Index not ready: {e}\n\n"
            "Run the setup pipeline first:\n"
            "```\n"
            "python src/fetch_jobs.py\n"
            "python src/clean_jobs.py\n"
            "python src/embed_jobs.py\n"
            "python src/build_faiss_index.py\n"
            "```"
        )
        st.stop()

    with st.spinner(f"Retrieving top {top_k} matches for **{preferred_role}**..."):
        # Local embed models don't need the Together client
        embed_model = get_models()["embed_model"]
        client_for_embed = None if embed_model.startswith("local:") else client
        matches = match_resume_to_jobs(
            resume_text, index, meta, clean_jobs, client_for_embed,
            preferred_role=preferred_role,
            top_k=top_k,
        )

    if not matches:
        st.warning("No matches found. Try selecting 'Any' role or re-fetching jobs.")
        st.stop()

    with st.spinner(f"Scoring top {top_n} matches with LLM..."):
        scored = score_top_jobs(resume_text, matches, client, top_n=top_n)

    st.success(f"Done! Showing top {len(scored)} results for **{preferred_role}**.")
    st.divider()

    # Keys are the python dict keys stored in scored results
    key_score   = "fit_score"
    key_matched = "matched_skills"
    key_missing = "missing_skills"
    key_recs    = "recommendations"
    key_summary = "summary"

    for job in scored:
        score   = job.get(key_score, 0)
        css_cls = score_color_class(score)
        tags_html = "".join(
            f'<span class="tag-pill">{t}</span>'
            for t in (job.get("tags") or [])[:8]
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### #{job['rank']} — {job['title']}")
            st.markdown(f"🏢 **{job['company']}** &nbsp;|&nbsp; 📍 {job['location'] or 'Remote'}")
            if tags_html:
                st.markdown(tags_html, unsafe_allow_html=True)
        with col2:
            st.markdown(
                f'<div style="text-align:center">'
                f'<span class="{css_cls}">{score}</span>'
                f'<br><small style="color:#888">/ 100 fit score</small>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with st.expander("📊 See full analysis", expanded=(job["rank"] == 1)):
            a, b = st.columns(2)
            with a:
                st.markdown("**Matched Skills**")
                matched = job.get(key_matched) or []
                if matched:
                    st.markdown(
                        " ".join(f'<span class="tag-pill skill-matched">{s}</span>' for s in matched),
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("None identified")
            with b:
                st.markdown("**Missing Skills**")
                missing = job.get(key_missing) or []
                if missing:
                    st.markdown(
                        " ".join(f'<span class="tag-pill skill-missing">{s}</span>' for s in missing),
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("None identified")

            st.markdown("**Recommendations**")
            recs = job.get(key_recs) or []
            if recs:
                for r in recs:
                    st.markdown(f"- {r}")
            else:
                st.caption("No recommendations.")

            st.info(f"**Summary:** {job.get(key_summary, 'N/A')}")
            st.markdown(f"[Apply here]({job['url']})")

        st.divider()

elif not uploaded:
    st.info("Upload your resume above to get started.")