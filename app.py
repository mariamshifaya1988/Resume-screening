

import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Session state
if "shortlisted" not in st.session_state:
    st.session_state.shortlisted = []

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Load model
model = pickle.load(open("resume_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# -----------------------
# Text Cleaning
# -----------------------
def clean_text(text):

    text = text.lower()
    text = re.sub(r"http\S+","",text)
    text = re.sub(r"[^a-zA-Z ]"," ",text)

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)

# -----------------------
# Role Prediction
# -----------------------
def predict_role(text):

    text = clean_text(text)

    vec = vectorizer.transform([text])

    role = model.predict(vec)[0]

    return role

# -----------------------
# Email Extraction
# -----------------------
def extract_email(text):

    email = re.findall(r"\S+@\S+",text)

    return email[0] if email else "Not Found"

# -----------------------
# Phone Extraction
# -----------------------
def extract_phone(text):

    phones = re.findall(r'\+?\d[\d\s\-]{8,15}\d', text)

    if phones:
        phone = phones[0]
        phone = re.sub(r"[^\d]","",phone)

        if len(phone) > 10:
            phone = phone[-10:]

        return phone

    return "Not Found"

# -----------------------
# Name Extraction
# -----------------------
def extract_name(text):

    lines = text.split("\n")

    for line in lines[:5]:

        if "name" in line.lower():
            return line.split(":")[-1].strip()

    return "Not Found"

# -----------------------
# Experience Extraction
# -----------------------
def extract_experience(text):

    exp = re.findall(r"\d+\s+years",text.lower())

    if exp:
        return int(exp[0].split()[0])

    return 0

# -----------------------
# Skill Extraction
# -----------------------
def extract_skills(text):

    skills_list = [
        "python","sql","machine learning","deep learning",
        "power bi","tableau","excel","nlp","pandas","numpy"
    ]

    found = []

    text = text.lower()

    for skill in skills_list:

        if skill in text:
            found.append(skill)

    return found


# -----------------------
# Streamlit UI
# -----------------------

st.title("AI Resume Screening System")

role = st.selectbox(
    "Select Required Role",
    ["Data Scientist","Data Analyst","Software Engineer","ML Engineer"]
)

skills = st.multiselect(
    "Required Skills",
    ["python","sql","machine learning","deep learning",
     "power bi","tableau","excel","nlp","pandas","numpy"]
)

experience = st.slider("Minimum Years of Experience",0,10)

files = st.file_uploader(
    "Upload resumes",
    type=["txt"],
    accept_multiple_files=True
)

# -----------------------
# Resume Screening
# -----------------------

if st.button("Screen Resumes"):

    st.session_state.shortlisted = []

    if files:

        for file in files:

            text = file.read().decode("utf-8")

            name = extract_name(text)
            email = extract_email(text)
            phone = extract_phone(text)

            predicted_role = predict_role(text)

            candidate_skills = extract_skills(text)

            exp = extract_experience(text)

            skill_match = all(skill in candidate_skills for skill in skills)

            if predicted_role == role and skill_match and exp >= experience:

                st.session_state.shortlisted.append({

                    "Name": name,
                    "Role": predicted_role,
                    "Skills": ", ".join(candidate_skills),
                    "Experience": exp,
                    "Email": email,
                    "Phone": phone,
                    "File Name": file.name,
                    "Resume Text": text

                })

# -----------------------
# Show Results
# -----------------------

if len(st.session_state.shortlisted) == 0:

    st.warning("No candidates shortlisted")

else:

    st.success(f"{len(st.session_state.shortlisted)} candidates shortlisted")

    df = pd.DataFrame(st.session_state.shortlisted)

    st.dataframe(df[["Name","Role","Skills","Experience","Email","Phone"]])

    st.markdown("### Download Resumes")

    for candidate in st.session_state.shortlisted:

        st.download_button(
            label=f"Download {candidate['Name']} Resume",
            data=candidate["Resume Text"],
            file_name=candidate["File Name"],
            key=candidate["File Name"]
        )

        st.markdown("---")
        
