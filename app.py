
import streamlit as st
import pickle
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# ==========================
# Load Model
# ==========================

model = pickle.load(open("resume_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# ==========================
# Clean Text
# ==========================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)

# ==========================
# Prediction Function
# ==========================

def predict_resume(resume_text):

    cleaned = clean_text(resume_text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    return prediction


# ==========================
# Streamlit UI
# ==========================

st.title("AI Resume Screening System")

st.write("Upload resumes and check if candidates are shortlisted.")

uploaded_files = st.file_uploader(
    "Upload Resume Files",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files:

    results = []

    for file in uploaded_files:

        resume_text = file.read().decode("utf-8")

        prediction = predict_resume(resume_text)

        status = "Shortlisted" if prediction == 1 else "Not Shortlisted"

        results.append({
            "File Name": file.name,
            "Prediction": status
        })

    df = pd.DataFrame(results)

    st.subheader("Screening Results")

    st.table(df)
