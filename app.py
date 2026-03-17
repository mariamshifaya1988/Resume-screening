

import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words=set(stopwords.words("english"))

model=pickle.load(open("resume_model.pkl","rb"))
vectorizer=pickle.load(open("vectorizer.pkl","rb"))

def clean_text(text):

    text=text.lower()
    text=re.sub(r"http\S+","",text)
    text=re.sub(r"[^a-zA-Z ]"," ",text)

    words=text.split()
    words=[w for w in words if w not in stop_words]

    return " ".join(words)

def predict_role(text):

    text=clean_text(text)

    vec=vectorizer.transform([text])

    role=model.predict(vec)[0]

    return role

def extract_email(text):

    email=re.findall(r"\S+@\S+",text)

    return email[0] if email else "Not Found"

def extract_phone(text):

    phone=re.findall(r"\d{10}",text)

    return phone[0] if phone else "Not Found"

def extract_name(text):

    lines = text.split("\n")

    for line in lines[:5]:

        if "name" in line.lower():

            return line.split(":")[-1].strip()

    return "Not Found"

def extract_experience(text):

    exp=re.findall(r"\d+\s+years",text.lower())

    if exp:
        return int(exp[0].split()[0])

    return 0

def extract_skills(text):

    skills_list=["python","sql","machine learning","deep learning",
                 "power bi","tableau","excel","nlp","pandas","numpy"]

    found=[]

    text=text.lower()

    for skill in skills_list:

        if skill in text:

            found.append(skill)

    return found

st.title("AI Resume Screening System")

role=st.selectbox("Select Required Role",
                  ["Data Scientist","Data Analyst",
                   "Software Engineer","ML Engineer"])

skills=st.multiselect("Required Skills",
                      ["python","sql","machine learning",
                       "deep learning","power bi","tableau",
                       "excel","nlp","pandas","numpy"])

experience=st.slider("Minimum Years of Experience",0,10)

files=st.file_uploader("Upload resumes",
                       type=["txt"],
                       accept_multiple_files=True)

if st.button("Screen Resumes"):

    results=[]

    for file in files:

        text=file.read().decode("utf-8")

        name=extract_name(text)
        email=extract_email(text)
        phone=extract_phone(text)

        predicted_role=predict_role(text)

        candidate_skills=extract_skills(text)

        exp=extract_experience(text)

        skill_match=all(skill in candidate_skills for skill in skills)

        if predicted_role==role and skill_match and exp>=experience:

            status="Shortlisted"

        else:

            status="Not Shortlisted"

        results.append({

            "Name":name,
            "File":file.name,
            "Role":predicted_role,
            "Skills":",".join(candidate_skills),
            "Experience":exp,
            "Email":email,
            "Phone":phone,
            "Prediction":status

        })

    df=pd.DataFrame(results)

    st.dataframe(df)
