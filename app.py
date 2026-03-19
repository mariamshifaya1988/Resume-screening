import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# UPDATED IMPORTS
import docx
import pdfplumber

# Session state
if "shortlisted" not in st.session_state:
    st.session_state.shortlisted = []

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Load model
model = pickle.load(open("resume_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# -----------------------
# File Reading Function (FIXED)
# -----------------------
def read_file(file):

    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])

    elif file.name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
        return text

    return ""

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
        phone = re.sub(r"[^\d]","",phones[0])
        return phone[-10:]

    return "Not Found"

# -----------------------
# Name Extraction (IMPROVED)
# -----------------------
def extract_name(text):

    lines = text.split("\n")

    for line in lines[:10]:
        line = line.strip()

        if len(line.split()) <= 4 and line.replace(" ", "").isalpha():
            return line

    return "Not Found"

# -----------------------
# Experience Extraction (IMPROVED)
# -----------------------
def extract_experience(text):

    matches = re.findall(r'(\d+)\+?\s*(years|yrs)', text.lower())

    if matches:
        return max([int(m[0]) for m in matches])

    return 0

# -----------------------
# Skill Extraction
# -----------------------
def extract_skills(text):

    skills_list = [
        "python","sql","machine learning","deep learning","system design","java","algorithm","data structure",
        "power bi","tableau","excel","nlp","pandas","numpy","tensorflow","computer vision","business analysis",
        "requirement gathering","product strategy","roadmap","agile","stakeholder management","aws","kubernetes",
        "linux","ci cd","docker","terraform","scikit-learn","cloud architecture","azure","data visualization",
        "data pipelines","spark","etl","hadoop","html","javascript","react","css","mongodb","nodejs",
        "talent management","hr policies","recruitment","employee engagement","model deployment","pytorch",
        "mlops","postgresql","flask","django","restapi","testing","selenium","test case","automation testing",
        "sales strategy","client management","crm","lead generation"
    ]

    found = []
    text = text.lower()

    for skill in skills_list:
        if skill in text:
            found.append(skill)

    return found

# -----------------------
# UI
# -----------------------
st.title("AI Resume Screening System")

role = st.selectbox("Select Required Role", [
    "Data Scientist","Data Analyst","Software Engineer","Machine Learning Engineer","AI Engineer",
    "Business Analyst","Cloud Engineer","Data Engineer","DevOps Engineer","Full Stack Developer",
    "HR Manager","Product Manager","Python Developer","QA Engineer","Sales Manager"
])

skills = st.multiselect("Required Skills", [
    "python","sql","machine learning","deep learning","java","power bi","tableau","excel",
    "nlp","pandas","numpy","tensorflow","aws","docker","kubernetes","flask","django"
])

experience = st.slider("Minimum Years of Experience",0,10)

files = st.file_uploader(
    "Upload resumes",
    type=["txt","pdf","docx"],
    accept_multiple_files=True
)

# -----------------------
# Screening
# -----------------------
if st.button("Screen Resumes"):

    st.session_state.shortlisted = []

    if files:

        for file in files:

            text = read_file(file)

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
                    "File Data": file.getvalue()   # ✅ FIXED
                })

# -----------------------
# Results
# -----------------------
if len(st.session_state.shortlisted) == 0:

    st.warning("No candidates shortlisted")

else:

    st.success(f"{len(st.session_state.shortlisted)} candidates shortlisted")

    headers = st.columns(7)
    for col, h in zip(headers, ["Name","Role","Skills","Exp","Email","Phone","Resume"]):
        col.write(h)

    st.markdown("---")

    for i, candidate in enumerate(st.session_state.shortlisted):

        cols = st.columns(7)

        cols[0].write(candidate["Name"])
        cols[1].write(candidate["Role"])
        cols[2].write(candidate["Skills"])
        cols[3].write(candidate["Experience"])
        cols[4].write(candidate["Email"])
        cols[5].write(candidate["Phone"])

        cols[6].download_button(
            label="Download",
            data=candidate["File Data"],  # ✅ REAL FILE
            file_name=candidate["File Name"],
            mime="application/octet-stream",
            key=f"download_{i}"
        )

        st.markdown("---")
