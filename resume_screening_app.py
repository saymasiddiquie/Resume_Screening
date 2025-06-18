# Force Streamlit to rebuild

import streamlit as st
import re
import PyPDF2
import docx2txt
import joblib
import numpy as np

label_mapping = {
    0: "Java Developer",
    1: "Database",
    2: "HR",
    3: "Data Science",
    4: "Advocate",
    5: "Automation Testing",
    6: "DevOps Engineer",
    7: "Testing",
    8: "DotNet Developer",
    9: "Hadoop",
    10: "SAP Developer",
    11: "Python Developer",
    12: "Health and fitness",
    13: "Civil Engineer",
    14: "Arts",
    15: "Business Analyst",
    16: "Web Designing",
    17: "Mechanical Engineer",
    18: "Sales",
    19: "ETL Developer",
    20: "Electrical Engineering",
    21: "Blockchain",
    22: "Network Security Engineer",
    23: "Operations Manager",
    24: "PMO"
}


# Load pre-trained model and vectorizer
model = joblib.load("resume_model.pkl")  # Ensure this file is in the same directory
vectorizer = joblib.load("vectorizer.pkl")  # Ensure this file is in the same directory

def clean_resume(resume_text):
    resume_text = re.sub(r'http\S+\s*', ' ', resume_text)
    resume_text = re.sub(r'#\S+', '', resume_text)
    resume_text = re.sub(r'@\S+', ' ', resume_text)
    resume_text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)
    resume_text = re.sub(r'\s+', ' ', resume_text)
    return resume_text

def extract_text(file):
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        return "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    else:
        return None

# Streamlit UI
st.set_page_config(page_title="Resume Screening App", layout="centered")
st.title("📄 AI-Powered Resume Screening System")

uploaded_file = st.file_uploader("Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    raw_text = extract_text(uploaded_file)
    if raw_text:
        st.subheader("Extracted Resume Text")
        st.text_area("", raw_text, height=250)

        cleaned_text = clean_resume(raw_text)

        if st.button("🔍 Predict Resume Category"):
            vec = vectorizer.transform([cleaned_text])
            prediction = model.predict(vec)
            st.success(f"🏷️ Predicted Category: **{prediction[0]}**")
    else:
        st.error("Failed to extract text from the uploaded file.")
else:
    st.info("Please upload a resume to get started.")

import joblib

clf = joblib.load("resume_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
resume_text = st.text_area("Paste the resume text here:")
vectorized_input = vectorizer.transform([resume_text])



prediction = clf.predict(vectorized_input)[0]
predicted_role = label_mapping.get(prediction, "Unknown Category")

st.write("### 📊 Resume Category Mapping")
for key, value in label_mapping.items():
    st.write(f"**{key}** : {value} 💼")

st.success(f"✅ The resume is classified as: **{predicted_role}** 👍")
st.snow()








