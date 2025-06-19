
import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
df = pd.read_csv("resume_dataset.csv", encoding="utf-8")

# Clean resume text safely
def clean_resume(text):
    text = text.replace('\x00', '')  # Remove null characters
    text = re.sub(r"http\S+\s*", " ", text)
    text = re.sub(r"RT|cc", " ", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"[%s]" % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

df["cleaned_resume"] = df["Resume"].apply(lambda x: clean_resume(str(x)))

# Encode labels
le = LabelEncoder()
df["Category_encoded"] = le.fit_transform(df["Category"])
y = df["Category_encoded"]

# Vectorize
tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_features=1500)
X = tfidf.fit_transform(df["cleaned_resume"])

# Oversample to balance
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=0
)

# Train model
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model, vectorizer, and encoder
joblib.dump(model, "resume_model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Model, vectorizer, and label encoder saved.")
