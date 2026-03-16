

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# ==========================
# Load Model
# ==========================

model = pickle.load(open("resume_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# ==========================
# Text Cleaning Function
# ==========================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
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

st.write("Upload or paste a resume to check if the candidate is shortlisted.")

resume_input = st.text_area("Paste Resume Text Here")

if st.button("Predict"):

    if resume_input.strip() == "":
        st.warning("Please enter resume text")

    else:

        result = predict_resume(resume_input)

        if result == 1:
            st.success("Candidate Likely Shortlisted")
        else:
            st.error("Candidate Not Shortlisted")
