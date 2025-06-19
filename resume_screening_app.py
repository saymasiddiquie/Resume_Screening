
import streamlit as st
import joblib
import re
import PyPDF2
import docx2txt

# Load saved model, vectorizer, and label encoder
model = joblib.load("resume_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Minimal cleaning
def clean_resume(text):
    return text.lower().strip()

# Extract resume content
def extract_text(file):
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        return "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    return ""

# Streamlit UI
st.set_page_config(page_title="Resume Screening App", layout="centered")
st.title("📄 AI-Powered Resume Screening System")

uploaded_file = st.file_uploader("Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    raw_text = extract_text(uploaded_file)
    if raw_text:
        st.subheader("Extracted Resume Text")
        st.text_area("Resume Preview", raw_text, height=250, label_visibility="collapsed")

        cleaned_text = clean_resume(raw_text)

        # Only transform and predict when button is clicked
        if st.button("🔍 Predict Resume Category"):
            vec = vectorizer.transform([cleaned_text])
            if vec.nnz == 0:
                st.error("❌ No recognizable keywords found in the resume.")
            else:
                prediction = model.predict(vec)[0]
                predicted_role = label_encoder.inverse_transform([prediction])[0]

                # Output
                st.success(f"✅ The resume is classified as: **{predicted_role}** 👍")
                st.snow()
    else:
        st.error("Failed to extract text from the uploaded file.")
else:
    st.info("📂 Please upload a resume to get started.")





