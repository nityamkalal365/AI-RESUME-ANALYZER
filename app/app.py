import streamlit as st
import pickle
import re
import nltk
import pandas as pd

nltk.download("stopwords")

from nltk.corpus import stopwords

# Load saved model and TF-IDF
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "src", "model.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "..", "src", "tfidf.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(TFIDF_PATH, "rb") as f:
    tfidf = pickle.load(f)


stop_words = set(stopwords.words("english"))

import re
from collections import Counter

# keep or expand your skill list
skills_list = [
    # Programming Languages
    "Python", "Java", "C++", "C", "JavaScript", "TypeScript",
    "SQL", "MATLAB", "R", "Bash", "Go", "PHP",

    # Machine Learning & AI Libraries
    "TensorFlow", "PyTorch", "Keras", "Scikit-learn",
    "XGBoost", "LightGBM", "CatBoost",
    "OpenCV", "NLTK", "spaCy", "Transformers",
    "Hugging Face", "Gensim", "MediaPipe",

    # Data Science & Analysis
    "Pandas", "NumPy", "SciPy",
    "Matplotlib", "Seaborn", "Plotly",
    "Statsmodels", "Power BI", "Tableau",
    "Excel", "Jupyter Notebook", "Google Colab",

    # Web Development & Backend
    "HTML", "CSS", "Bootstrap", "Tailwind CSS",
    "React", "Angular", "Vue",
    "Node.js", "Express.js", "Flask", "Django",
    "Spring Boot", "REST API", "GraphQL",

    # Databases
    "MySQL", "PostgreSQL", "MongoDB",
    "SQLite", "Redis", "Firebase", "Oracle DB",

    # DevOps & Cloud
    "AWS", "Azure", "Google Cloud",
    "Docker", "Kubernetes", "Jenkins",
    "GitHub Actions", "Terraform", "Nginx",

    # Software Engineering Tools
    "Git", "GitHub", "GitLab",
    "Linux", "Bash Scripting", "Unit Testing", "PyTest",
    "System Design", "Microservices", "CI/CD"

]

# optional synonyms map (map common variations to canonical skill)
SYNONYMS = {
    "ml": "machine learning",
    "machine-learning": "machine learning",
    "deep-learning": "deep learning",
    "tf": "tensorflow",
    "tensor flow": "tensorflow",
    "pytorch": "pytorch",
    "js": "javascript",
    "js ": "javascript"
}


def normalize_text_for_matching(text: str) -> str:
    """Lowercase and normalize whitespace/punctuation for better matching."""
    text = text.lower()
    # replace punctuation with space
    text = re.sub(r"[^\w\s]", " ", text)
    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def variants(skill: str):
    """Return variants for a skill to improve matching (no-space variant, simple synonyms)."""
    s = skill.lower()
    v = {s}
    v.add(s.replace(" ", ""))       # machinelearning
    v.add(s.replace(" ", "-"))      # machine-learning
    return v


def extract_skills_counts(text: str) -> dict:
    """
    Returns a dict {skill: count} found in text.
    Uses exact word/phrase matches plus simple normalization variants and synonyms.
    """
    if not isinstance(text, str) or not text.strip():
        return {}

    text_norm = normalize_text_for_matching(text)

    # expand synonyms into the text to catch 'ml' -> 'machine learning'
    for k, v in SYNONYMS.items():
        # replace synonym occurrences with canonical skill to help counting
        text_norm = re.sub(r"\b" + re.escape(k) + r"\b", v, text_norm)

    counts = Counter()

    # Count occurrences of each skill (with variants)
    for skill in skills_list:
        for variant in variants(skill):
            # word boundary match for phrase or single token
            pattern = r"\b" + re.escape(variant) + r"\b"
            found = re.findall(pattern, text_norm)
            if found:
                counts[skill] += len(found)

    # As a fallback, check tokens (for very short keywords)
    # This catches 'numpy' even if punctuation/formatting different
    tokens = text_norm.split()
    for token in tokens:
        for skill in skills_list:
            if token == skill and counts[skill] == 0:
                counts[skill] += 1

    # Remove zero-count skills
    return {k: v for k, v in counts.items() if v > 0}

# ---------- Functions ----------

def clean_text(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def predict_category_with_confidence(resume_text):
    cleaned = clean_text(resume_text)
    vectorized = tfidf.transform([cleaned])

    probs = model.predict_proba(vectorized)[0]
    max_index = probs.argmax()

    category = model.classes_[max_index]
    confidence = probs[max_index] * 100

    return category, round(confidence, 2)





def extract_skills_from_jd(jd_text: str) -> dict:
    """
    Extract skills from job description. We treat JD skills as required once each,
    but we normalize same as resume to detect phrases.
    """
    if not jd_text or not jd_text.strip():
        return {}
    jd_norm = normalize_text_for_matching(jd_text)
    # map synonyms in JD too
    for k, v in SYNONYMS.items():
        jd_norm = re.sub(r"\b" + re.escape(k) + r"\b", v, jd_norm)

    jd_counts = Counter()
    for skill in skills_list:
        for variant in variants(skill):
            pattern = r"\b" + re.escape(variant) + r"\b"
            found = re.findall(pattern, jd_norm)
            if found:
                jd_counts[skill] += len(found)
    # if JD has zero counts we try token-based (rare)
    return {k: v for k, v in jd_counts.items() if v > 0}


def ats_score_with_counts(resume_text: str, jd_text: str):
    """
    Returns (score_percent, resume_skill_counts, jd_skill_counts, matched_detail)
    Score logic:
      - base_match = number_of_unique_required_skills_found / number_required_skills
      - bonus for multiple mentions: min(total_found_mentions, number_required_skills) / (2*number_required_skills)
      - final_score = (base_match + bonus) * 100, capped at 100
    """
    resume_skills = extract_skills_counts(resume_text)
    jd_skills = extract_skills_from_jd(jd_text)

    if len(jd_skills) == 0:
        # nothing required in JD -> can't compute meaningful ATS
        return 0.0, resume_skills, jd_skills, {}

    required = set(jd_skills.keys())
    found = set(resume_skills.keys()) & required

    if len(required) == 0:
        return 0.0, resume_skills, jd_skills, {}

    base_match = len(found) / len(required)

    # count total matched mentions (sum counts of matched skills from resume)
    total_matched_mentions = sum(resume_skills[s] for s in found)

    # bonus scaled so repeated mentions give up to +50% of base_match
    bonus = min(total_matched_mentions, len(required)) / (2 * len(required))

    score = (base_match + bonus) * 100.0
    score = min(100.0, round(score, 2))

    # detail per skill for UI: (skill, resume_count, jd_count, matched_bool)
    matched_detail = {
        skill: {
            "resume_count": resume_skills.get(skill, 0),
            "jd_count": jd_skills.get(skill, 0),
            "matched": skill in found
        }
        for skill in required
    }

    return score, resume_skills, jd_skills, matched_detail


# ---------- Streamlit UI ----------

st.title("📄 AI Resume Analyzer")
st.write("Upload resume text and job description to get ATS score and predicted role.")

resume_text = st.text_area("Paste Resume Text")
job_desc = st.text_area("Paste Job Description")

if st.button("Analyze Resume"):

    if resume_text.strip() == "":
        st.warning("⚠️ Please paste resume text to analyze.")
    
    else:
        # --- Prediction ---
        category, confidence = predict_category_with_confidence(resume_text)

        st.subheader("🔹 Predicted Job Category")
        st.success(category)

        st.subheader("📊 Prediction Confidence")
        st.write(f"{confidence}%")

        if confidence < 60:
            st.warning("⚠️ Low confidence prediction. Resume may be unclear or unrelated.")

        # --- ATS only if JD provided ---
        if job_desc.strip() == "":
            st.info("ℹ️ Add a job description to compute ATS match score.")
        
        else:
           
            score, resume_counts, jd_counts, matched_detail = ats_score_with_counts(resume_text, job_desc)
            
            # Resume skills table
            st.subheader("🧠 Extracted Resume Skills (skill : count)")
            if resume_counts:
                df_resume = pd.DataFrame(
                    sorted(resume_counts.items(), key=lambda x: -x[1]),
                    columns=["Skill", "Count"]
                )
                st.table(df_resume)
            else:
                st.write([])
            
            # JD skills table
            st.subheader("📌 Job Description Skills (required → found)")
            if jd_counts:
                jd_rows = []
                for skill, jd_c in jd_counts.items():
                    rd = matched_detail.get(skill, {})
                    jd_rows.append({
                        "Skill": skill,
                        "Required in JD": jd_c,
                        "Found in Resume": rd.get("resume_count", 0),
                        "Matched": rd.get("matched", False)
                    })
                df_jd = pd.DataFrame(jd_rows)
                st.table(df_jd)
            else:
                st.write([])
            
            # ATS score
            st.subheader("📊 ATS Match Score")
            st.info(f"{score}% match with job description")

