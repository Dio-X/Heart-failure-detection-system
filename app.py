import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Heart Risk CDSS", page_icon="🫀", layout="centered")

# --- 1. LOAD THE TRAINED MODEL (SRS FR-03) ---
@st.cache_resource
def load_model():
    try:
        # Load the dictionary containing the model and feature names
        model_data = joblib.load('heart_model.pkl')
        return model_data['model'], model_data['feature_names']
    except FileNotFoundError:
        st.error("Error: 'heart_model.pkl' not found. Please run 'train_model.py' first.")
        return None, None

model, feature_names = load_model()

# --- 2. DEFINE INPUT ATTRIBUTES (SRS FR-01: Modular Config) ---
# This is the "Flexible" part. We define the logic here, and the UI generates itself.
# Format: "Internal Name": {"Label": "Display Name", "Type": "Number/Select", "Min": 0, "Max": 100, "Options": []}
attribute_config = {
    "age": {"label": "Age (Years)", "type": "number", "min": 1, "max": 120, "default": 50},
    "sex": {"label": "Sex", "type": "select", "options": ["Male", "Female"]},
    "cp": {"label": "Chest Pain Type", "type": "select", "options": ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]},
    "trestbps": {"label": "Resting Blood Pressure (mm Hg)", "type": "number", "min": 80, "max": 200, "default": 120},
    "chol": {"label": "Serum Cholesterol (mg/dl)", "type": "number", "min": 100, "max": 600, "default": 200},
    "fbs": {"label": "Fasting Blood Sugar > 120 mg/dl", "type": "select", "options": ["False", "True"]},
    "restecg": {"label": "Resting ECG Results", "type": "select", "options": ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]},
    "thalach": {"label": "Max Heart Rate Achieved", "type": "number", "min": 60, "max": 220, "default": 150},
    "exang": {"label": "Exercise Induced Angina", "type": "select", "options": ["No", "Yes"]},
    "oldpeak": {"label": "ST Depression (Oldpeak)", "type": "number", "min": 0.0, "max": 6.0, "default": 1.0, "step": 0.1},
    "slope": {"label": "Slope of Peak Exercise ST", "type": "select", "options": ["Upsloping", "Flat", "Downsloping"]},
    "ca": {"label": "Number of Major Vessels (0-3)", "type": "number", "min": 0, "max": 3, "default": 0},
    "thal": {"label": "Thalassemia", "type": "select", "options": ["Normal", "Fixed Defect", "Reversible Defect"]}
}

# --- 3. UI HEADER ---
st.title("🫀 Heart Failure Risk Prediction")
st.markdown("Enter patient clinical data below to assess heart disease risk using the **Cleveland Standard** protocol.")

if model:
    # --- 4. DYNAMIC FORM GENERATION (SRS FR-02) ---
    user_inputs = {}
    
    # Create two columns for a cleaner layout
    col1, col2 = st.columns(2)
    keys = list(attribute_config.keys())
    
    # split inputs between columns
    half = len(keys) // 2
    
    for i, key in enumerate(keys):
        # Select column
        col = col1 if i < half else col2
        config = attribute_config[key]
        
        with col:
            if config["type"] == "number":
                user_inputs[key] = st.number_input(
                    config["label"], 
                    min_value=config["min"], 
                    max_value=config["max"], 
                    value=config["default"],
                    step=config.get("step", 1)
                )
            elif config["type"] == "select":
                display_val = st.selectbox(config["label"], config["options"])
                
                # Convert text back to numbers for the model
                # Note: This mapping matches the standard Cleveland encoding
                if key == "sex":
                    user_inputs[key] = 1 if display_val == "Male" else 0
                elif key == "fbs" or key == "exang":
                    user_inputs[key] = 1 if display_val == "True" or display_val == "Yes" else 0
                elif key == "cp":
                    user_inputs[key] = config["options"].index(display_val)
                elif key == "restecg":
                    user_inputs[key] = config["options"].index(display_val)
                elif key == "slope":
                    user_inputs[key] = config["options"].index(display_val)
                elif key == "thal":
                    # Thalassemia usually mapped: 3=Normal, 6=Fixed, 7=Reversible in some datasets, 
                    # but commonly 0,1,2 or 1,2,3 in processed sets.
                    # We use simple index mapping 0,1,2 for simplicity with this specific processed dataset.
                    user_inputs[key] = config["options"].index(display_val)

    st.markdown("---")

    # --- 5. PREDICTION LOGIC (SRS FR-03 & FR-04) ---
    if st.button("Analyze Risk", type="primary", use_container_width=True):
        # Prepare data frame for model (Must match feature names exactly)
        input_df = pd.DataFrame([user_inputs])
        
        # Make Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] # Probability of Class 1 (Disease)
        
        st.subheader("Assessment Result:")
        
        if prediction == 1:
            st.error(f"⚠️ **HIGH RISK DETECTED**")
            st.markdown(f"The model predicts a **{probability*100:.1f}%** probability of heart disease.")
            st.warning("Recommendation: Consult a cardiologist for further testing.")
        else:
            st.success(f"✅ **LOW RISK / NORMAL**")
            st.markdown(f"The model predicts a **{(1-probability)*100:.1f}%** probability of being healthy.")