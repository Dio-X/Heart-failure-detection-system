import pandas as pd
import io
import requests

# Authentic URL from the UCI Machine Learning Repository mirror
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# The UCI data doesn't have headers, so we define them manually (The 13 attributes + target)
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

try:
    response = requests.get(url)
    if response.status_code == 200:
        # Load data, handling '?' which represents missing values in this dataset
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')), names=column_names, na_values="?")

        # Drop rows with missing values (clean the data instantly)
        df = df.dropna()

        # The original dataset has target 0-4. We convert 1,2,3,4 to 1 (Disease) for binary classification
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

        # Save to CSV
        df.to_csv('heart.csv', index=False)
        print("✅ Success! 'heart.csv' has been created.")
        print(f"Dataset shape: {df.shape}")
        print(df.head())
    else:
        print("Failed to download data from UCI.")
except Exception as e:
    print(f"Error: {e}")