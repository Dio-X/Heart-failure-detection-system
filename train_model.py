import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # Used to save the model



df = pd.read_csv('heart.csv')

print("Loading Dataset...")
print(f"Total Patients: {len(df)}")



X = df.drop('target', axis=1)  # The 13 Inputs
y = df['target']               # The Answer


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Training Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")



model_data = {
    'model': model,
    'feature_names': X.columns.tolist()
}

joblib.dump(model_data, 'heart_model.pkl')
print("Model saved as 'heart_model.pkl'")