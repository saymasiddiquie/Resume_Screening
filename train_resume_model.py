
import pandas as pd
import numpy as np
import re
import nltk
import string
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv("C:/Users/Sayama Siddiquie/Downloads/Resume_Screening_System/resume_dataset.csv", encoding="utf-8")

# Cleaning function
def clean_resume(text):
    text = re.sub(r"http\S+\s*", " ", text)
    text = re.sub(r"RT|cc", " ", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"[%s]" % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), " ", text)
    text = re.sub(r"[^\x00-\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

df["cleaned_resume"] = df["Resume"].apply(lambda x: clean_resume(str(x)))

# Encode labels
le = LabelEncoder()
df["Category"] = le.fit_transform(df["Category"])

# Vectorize text
tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_features=1500)
X = tfidf.fit_transform(df["cleaned_resume"].values)
y = df["Category"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
model = OneVsRestClassifier(KNeighborsClassifier())
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, "resume_model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully.")
