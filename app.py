

import streamlit as st
import pickle
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# ==========================
# Load Model and Vectorizer
# ==========================

model = pickle.load(open("resume_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# ==========================
# Text Cleaning Function
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

st.write("Upload resumes from your computer to check candidate shortlisting.")

# Upload button
uploaded_files = st.file_uploader(
    "Upload Resume Files",
    type=["txt"],
    accept_multiple_files=True
)

# ==========================
# Predict Button
# ==========================

if st.button("Predict"):

    if uploaded_files is None or len(uploaded_files) == 0:
        st.warning("Please upload at least one resume file.")

    else:

        results = []

        for file in uploaded_files:

            resume_text = file.read().decode("utf-8")

            prediction = predict_resume(resume_text)

            status = "Shortlisted" if prediction == 1 else "Not Shortlisted"

            results.append({
                "File Name": file.name,
                "Prediction": status
            })

        result_df = pd.DataFrame(results)

        st.subheader("Screening Results")

        st.table(result_df)    
