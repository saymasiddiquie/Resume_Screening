# 🤖 Resume Screening System using NLP and Machine Learning

This project is a **Streamlit-based web application** that classifies resumes into relevant job categories using **Natural Language Processing (NLP)** and **Machine Learning**. Upload your resume (PDF or DOCX), and the app intelligently predicts the job domain — like `Data Science`, `Java Developer`, `HR`, etc. — with instant feedback and visual animations. 🚀

---

## 📌 Key Features

- 📄 **Resume Upload**: Accepts resumes in `.pdf` and `.docx` formats.
- 🧠 **ML-based Job Role Prediction**: Uses a trained `KNeighborsClassifier` to categorize resumes.
- 📊 **Supports 25+ Categories**: From Python Developer to Sales, Civil Engineer to Blockchain Expert.
- 🎉 **Streamlit Frontend with Animations**: Thumbs up and balloons for positive user experience.
- 🧹 **Text Preprocessing**: Uses `nltk`, `re`, `string`, and `docx2txt` for cleaning and parsing.
- 🔍 **Real-time Predictions**: Instant feedback after file upload.

---

## 📂 Categories Supported

| Category                  | Count |
|--------------------------|-------|
| Java Developer           | 14    |
| Database                 | 11    |
| HR                       | 11    |
| Data Science             | 10    |
| Advocate                 | 10    |
| Automation Testing       | 7     |
| DevOps Engineer          | 7     |
| Testing                  | 7     |
| DotNet Developer         | 7     |
| Hadoop                   | 7     |
| SAP Developer            | 6     |
| Python Developer         | 6     |
| Health and Fitness       | 6     |
| Civil Engineer           | 6     |
| Arts                     | 6     |
| Business Analyst         | 6     |
| Web Designing            | 5     |
| Mechanical Engineer      | 5     |
| Sales                    | 5     |
| ETL Developer            | 5     |
| Electrical Engineering   | 5     |
| Blockchain               | 5     |
| Network Security Engg.   | 5     |
| Operations Manager       | 4     |
| PMO                      | 3     |

---

## 🛠️ Technologies Used

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python
- **ML Model**: scikit-learn (KNN classifier)
- **Libraries**:
  - `nltk` (for tokenization and cleaning)
  - `PyPDF2` (for reading PDFs)
  - `docx2txt` (for extracting text from Word files)
  - `joblib`, `pandas`, `numpy`

---

## 🚀 Getting Started (Run Locally)

### 1. Clone the Repository

git clone https://github.com/saymasiddiquie/Resume_Screening.git
cd Resume_Screening

Install Dependencies
pip install -r requirements.txt

Run the App
streamlit run resume_screening_app.py

Folder Structure
Resume_Screening/
│
├── resume_screening_app.py        # Streamlit app
├── train_resume_model.py          # Training script for the model
├── resume_model.pkl               # Trained model (generated)
├── requirements.txt               # Required packages

live app link
https://resumescreening-lfmwa4hybkudwtwhdhtvav.streamlit.app/


