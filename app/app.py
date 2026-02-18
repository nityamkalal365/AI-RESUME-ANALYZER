import streamlit as st
import pickle
import re
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

# Skill list
skills_list = [
    "python", "java", "sql", "machine learning", "deep learning",
    "tensorflow", "pytorch", "spring", "react", "node", "aws",
    "docker", "kubernetes", "data analysis", "pandas", "numpy",
    "excel", "communication", "leadership"
]

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


def extract_skills(text):
    text = text.lower()
    return list(set([skill for skill in skills_list if skill in text]))


def ats_score(resume_text, job_description):
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(job_description)

    if len(jd_skills) == 0:
        return 0, resume_skills, jd_skills

    match_count = len(set(resume_skills) & set(jd_skills))
    score = (match_count / len(jd_skills)) * 100

    return round(score, 2), resume_skills, jd_skills


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
            score, r_skills, j_skills = ats_score(resume_text, job_desc)

            st.subheader("🧠 Extracted Resume Skills")
            st.write(r_skills)

            st.subheader("📌 Job Description Skills")
            st.write(j_skills)

            st.subheader("📊 ATS Match Score")
            st.info(f"{score}% match with job description")
