import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load your resume dataset
df = pd.read_csv("resume_dataset.csv")

# Fit the LabelEncoder on the Category column
le = LabelEncoder()
le.fit(df["Category"])

# Save it for use in your app
joblib.dump(le, "label_encoder.pkl")

print("✅ label_encoder.pkl saved successfully.")
