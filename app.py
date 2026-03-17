
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
# Text Cleaning (NLP)
# ==========================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)

# ==========================
# Resume Prediction
# ==========================

def predict_resume(text):

    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    return prediction


# ==========================
# Extract Email
# ==========================

def extract_email(text):

    email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'

    emails = re.findall(email_pattern, text)

    return emails[0] if emails else "Not Found"


# ==========================
# Extract Phone Number
# ==========================

def extract_phone(text):

    phone_pattern = r'\d{10}'

    phones = re.findall(phone_pattern, text)

    return phones[0] if phones else "Not Found"


# ==========================
# Extract Designation
# ==========================

def extract_designation(text):

    roles = [
        "data scientist",
        "software engineer",
        "data analyst",
        "machine learning engineer",
        "hr manager",
        "project manager"
    ]

    text_lower = text.lower()

    for role in roles:
        if role in text_lower:
            return role.title()

    return "Not Found"


# ==========================
# Streamlit UI
# ==========================

st.title("AI Resume Screening System (NLP)")

st.write("Upload resumes and select the required job role.")

# Recruiter selects role
required_role = st.selectbox(
    "Select Required Role",
    [
        "Data Scientist",
        "Software Engineer",
        "Data Analyst",
        "Machine Learning Engineer",
        "HR Manager"
    ]
)

uploaded_files = st.file_uploader(
    "Upload Resume Files",
    type=["txt"],
    accept_multiple_files=True
)

# ==========================
# Prediction
# ==========================

if st.button("Screen Resumes"):

    if not uploaded_files:
        st.warning("Please upload resumes")

    else:

        results = []

        for file in uploaded_files:

            resume_text = file.read().decode("utf-8")

            prediction = predict_resume(resume_text)

            status = "Shortlisted" if prediction == 1 else "Not Shortlisted"

            email = extract_email(resume_text)
            phone = extract_phone(resume_text)
            designation = extract_designation(resume_text)

            if status == "Shortlisted":
                results.append({
                    "File Name": file.name,
                    "Prediction": status,
                    "Email": email,
                    "Phone": phone,
                    "Designation": designation
                })

            else:
                results.append({
                    "File Name": file.name,
                    "Prediction": status,
                    "Email": "-",
                    "Phone": "-",
                    "Designation": "-"
                })

        result_df = pd.DataFrame(results)

        st.subheader("Screening Results")

        st.table(result_df)  
