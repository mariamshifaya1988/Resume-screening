import streamlit as st
import pickle
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

import docx
import pdfplumber

# -----------------------
# Setup
# -----------------------
nltk.download("stopwords")

if "shortlisted" not in st.session_state:
    st.session_state.shortlisted = []

stop_words = set(stopwords.words("english"))

model = pickle.load(open("resume_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -----------------------
# Roles
# -----------------------
ROLES = [
    "AI Engineer","Business Analyst","Cloud Engineer","Data Analyst",
    "Data Engineer","Data Scientist","DevOps Engineer","Full Stack Developer",
    "HR Manager","Machine Learning Engineer","Product Manager","Python Developer",
    "QA Engineer","Sales Manager","Software Engineer"
]

# -----------------------
# Skills
# -----------------------
SKILLS = [
    "nlp","deep learning","python","tensorflow","computer vision","sql",
    "business analysis","excel","requirement gathering","power bi",
    "product strategy","roadmap","agile","stakeholder management","aws",
    "kubernetes","linux","ci cd","docker","terraform","machine learning",
    "scikit-learn","pandas","numpy","cloud architecture","azure","tableau",
    "data visualization","data pipelines","spark","etl","hadoop","html",
    "javascript","react","css","mongodb","node js","talent management",
    "hr policies","recruitment","employee engagement","model deployment",
    "pytorch","mlops","postgresql","flask","django","rest api","testing",
    "selenium","automation testing","test case","sales strategy",
    "client management","crm","lead generation","system design","java",
    "algorithm","data structure"
]

# -----------------------
# File Reading
# -----------------------
def read_file(file):

    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])

    elif file.name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
        return text

    return ""

# -----------------------
# Clean Text
# -----------------------
def clean_text(text):

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)

# -----------------------
# Role Prediction
# -----------------------
def predict_role(text):

    text_lower = text.lower()

    for r in ROLES:
        if r.lower() in text_lower:
            return r

    vec = vectorizer.transform([clean_text(text)])
    return model.predict(vec)[0]

# -----------------------
# Email
# -----------------------
def extract_email(text):

    match = re.findall(r"\S+@\S+", text)
    return match[0] if match else "Not Found"

# -----------------------
# Phone
# -----------------------
def extract_phone(text):

    match = re.findall(r'\+?\d[\d\s\-]{8,15}\d', text)

    if match:
        phone = re.sub(r"[^\d]", "", match[0])
        return phone[-10:]

    return "Not Found"

# -----------------------
# Name
# -----------------------
def extract_name(text):

    lines = text.split("\n")

    for line in lines[:15]:

        line = line.strip()

        if "name" in line.lower():
            parts = line.split(":")
            if len(parts) > 1:
                name = parts[1].strip()
                if len(name.split()) >= 2:
                    return name

        if line.lower() in ["resume","cv","curriculum vitae"]:
            continue

        if any(word in line.lower() for word in ["email","phone","summary","experience"]):
            continue

        if 2 <= len(line.split()) <= 4 and line.replace(" ","").isalpha():
            return line

    return "Not Found"

# -----------------------
# Experience
# -----------------------
def extract_experience(text):

    text = text.lower().replace("\n", " ")

    # Match formats like:
    # 3 years, 3 year, 3+ years, 3 yrs
    matches = re.findall(r'(\d+)\+?\s*(year|years|yr|yrs)', text)

    if matches:
        return max([int(m[0]) for m in matches])

    # Fallback: "experience: 3"
    match = re.search(r'experience[:\s]+(\d+)', text)
    if match:
        return int(match.group(1))

    return 0

# -----------------------
# Skills Extraction
# -----------------------
def extract_skills(text):

    text = text.lower()
    found = []

    for skill in SKILLS:
        if skill in text:
            found.append(skill)

    return found

# -----------------------
# UI
# -----------------------
st.title("AI Resume Screening System")

role = st.selectbox("Select Role", ROLES)

skills = st.multiselect("Required Skills", SKILLS)

experience = st.slider("Minimum Experience", 0, 10)

files = st.file_uploader(
    "Upload Resumes",
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
            role_pred = predict_role(text)
            exp = extract_experience(text)
            skill_list = extract_skills(text)

            skill_match = all(s in skill_list for s in skills)

            if role_pred == role and skill_match and exp >= experience:

                st.session_state.shortlisted.append({
                    "Name": name,
                    "Role": role_pred,
                    "Skills": ", ".join(skill_list),
                    "Experience": exp,
                    "Email": email,
                    "Phone": phone,
                    "File Name": file.name,
                    "File Data": file.getvalue()
                })

# -----------------------
# Results (CUSTOM TABLE)
# -----------------------
if len(st.session_state.shortlisted) == 0:

    st.warning("No candidates shortlisted")

else:

    st.success(f"{len(st.session_state.shortlisted)} candidates shortlisted")

    # Header
    h1, h2, h3, h4, h5, h6, h7 = st.columns(7)

    h1.write("**Name**")
    h2.write("**Role**")
    h3.write("**Skills**")
    h4.write("**Experience**")
    h5.write("**Email**")
    h6.write("**Phone**")
    h7.write("**Resume**")

    st.markdown("---")

    # Rows
    for i, candidate in enumerate(st.session_state.shortlisted):

        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

        c1.write(candidate["Name"])
        c2.write(candidate["Role"])
        c3.write(candidate["Skills"])
        c4.write(candidate["Experience"])
        c5.write(candidate["Email"])
        c6.write(candidate["Phone"])

        c7.download_button(
            label="Download",
            data=candidate["File Data"],
            file_name=candidate["File Name"],
            key=f"download_{i}"
        )

        st.markdown("---")
