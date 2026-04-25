import pandas as pd
import io
import requests


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

try:
    response = requests.get(url)
    if response.status_code == 200:

        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')), names=column_names, na_values="?")


        df = df.dropna()


        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)


        df.to_csv('heart.csv', index=False)
        print("✅ Success! 'heart.csv' has been created.")
        print(f"Dataset shape: {df.shape}")
        print(df.head())
    else:
        print("Failed to download data from UCI.")
except Exception as e:
    print(f"Error: {e}")