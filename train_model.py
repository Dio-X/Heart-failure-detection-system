import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # Used to save the model

# 1. Load the Dataset
# We use the Cleveland dataset we downloaded earlier
df = pd.read_csv('heart.csv')

print("Loading Dataset...")
print(f"Total Patients: {len(df)}")

# 2. Preprocessing (Renaming columns for clarity if needed, but we keep them standard)
# The dataset 'target' column: 1 = Disease, 0 = No Disease
X = df.drop('target', axis=1)  # The 13 Inputs
y = df['target']               # The Answer

# 3. Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model (Random Forest)
print("Training Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the Model AND the Column Names
# We save the column names so the App knows exactly what inputs to ask for (Flexibility!)
model_data = {
    'model': model,
    'feature_names': X.columns.tolist()
}

joblib.dump(model_data, 'heart_model.pkl')
print("Model saved as 'heart_model.pkl'")