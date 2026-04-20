import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# --- 1. ENHANCED PAGE UI ---
st.set_page_config(page_title="Heart Risk CDSS", page_icon="🫀", layout="wide")

# Custom CSS to make it look premium
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .main-header { font-size: 2.5rem; color: #1E3A8A; font-weight: 700; text-align: center; margin-bottom: 0;}
    .sub-header { font-size: 1.2rem; color: #64748B; text-align: center; margin-bottom: 2rem;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🫀 AI Heart Health Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Clinical Decision Support with Explainable AI</p>', unsafe_allow_html=True)

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    model_data = joblib.load('heart_model.pkl')
    return model_data['model'], model_data['feature_names']

model, feature_names = load_model()

# --- 3. LAYMAN MODE TOGGLE (UI Sidebar) ---
with st.sidebar:
    st.header("⚙️ Settings")
    layman_mode = st.toggle("Enable Layman Mode 🧑‍⚕️", value=True, help="Translates complex medical jargon into easy-to-understand English.")
    st.markdown("---")
    st.markdown("**How it works:**\nToggle this on if you are a patient. Turn it off if you are a medical professional entering clinical data.")

# --- 4. DYNAMIC DICTIONARIES (Medical vs Layman) ---
def get_label(medical_label, layman_label, layman_desc=""):
    if layman_mode:
        return f"{layman_label}", layman_desc
    return medical_label, ""

# --- 5. THE UI FORM (Categorized into Tabs for UX) ---
if model:
    tab1, tab2, tab3 = st.tabs(["👤 Basic Info", "🩺 Vitals & Lab Results", "⚠️ Symptoms & ECG"])
    
    user_inputs = {}

    # TAB 1: BASIC INFO
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            lbl, hlp = get_label("Age (Years)", "What is your age?", "Your age in years.")
            user_inputs['age'] = st.number_input(lbl, min_value=1, max_value=120, value=50, help=hlp)
        with col2:
            lbl, hlp = get_label("Sex", "Biological Sex", "Assigned at birth.")
            sex_choice = st.selectbox(lbl, ["Male", "Female"], help=hlp)
            user_inputs['sex'] = 1 if sex_choice == "Male" else 0

    # TAB 2: VITALS
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            lbl, hlp = get_label("trestbps (Resting BP)", "Resting Blood Pressure", "Normal is around 120/80. Enter the top number (Systolic).")
            user_inputs['trestbps'] = st.number_input(lbl, min_value=80, max_value=200, value=120, help=hlp)
            
            lbl, hlp = get_label("chol (Serum Cholesterol)", "Total Cholesterol Level", "Usually measured in a blood test. Normal is under 200.")
            user_inputs['chol'] = st.number_input(lbl, min_value=100, max_value=600, value=200, help=hlp)
        
        with col2:
            lbl, hlp = get_label("fbs > 120 mg/dl", "Is your fasting blood sugar high? (Signs of Diabetes)", "Is it over 120 mg/dl?")
            fbs_choice = st.selectbox(lbl, ["No", "Yes"], help=hlp)
            user_inputs['fbs'] = 1 if fbs_choice == "Yes" else 0
            
            lbl, hlp = get_label("ca (Major Vessels)", "Number of major blood vessels blocked", "Found via X-ray/fluoroscopy. Usually 0 for healthy.")
            user_inputs['ca'] = st.number_input(lbl, min_value=0, max_value=3, value=0, help=hlp)

    # TAB 3: SYMPTOMS & ECG
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            # CHEST PAIN - The Layman translation shines here
            cp_options = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
            layman_cp_options = [
                "1. Heavy pressure/squeezing during stress (Typical)", 
                "2. Sharp/stabbing pain but not typical (Atypical)", 
                "3. Muscle or stomach pain (Non-anginal)", 
                "4. No chest pain at all (Asymptomatic)"
            ]
            lbl, hlp = get_label("cp (Chest Pain Type)", "Describe your chest pain", "Select the one that best matches how your chest feels.")
            cp_choice = st.selectbox(lbl, layman_cp_options if layman_mode else cp_options, help=hlp)
            user_inputs['cp'] = layman_cp_options.index(cp_choice) if layman_mode else cp_options.index(cp_choice)

            lbl, hlp = get_label("exang (Exercise Angina)", "Does your chest hurt ONLY when you exercise?", "Pain triggered by walking/running.")
            exang_choice = st.selectbox(lbl, ["No", "Yes"], help=hlp)
            user_inputs['exang'] = 1 if exang_choice == "Yes" else 0
            
            lbl, hlp = get_label("thalach (Max HR)", "Maximum Heart Rate Achieved", "Highest heart rate during a treadmill test or workout.")
            user_inputs['thalach'] = st.number_input(lbl, min_value=60, max_value=220, value=150, help=hlp)

        with col2:
            lbl, hlp = get_label("restecg", "Resting ECG Scan Results", "0=Normal, 1=Wave Abnormality, 2=Hypertrophy")
            ecg_opts = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
            user_inputs['restecg'] = ecg_opts.index(st.selectbox(lbl, ecg_opts, help=hlp))

            lbl, hlp = get_label("oldpeak", "ECG ST Depression amount", "Value from an ECG reading (e.g., 1.5).")
            user_inputs['oldpeak'] = st.number_input(lbl, min_value=0.0, max_value=6.0, value=1.0, step=0.1, help=hlp)

            lbl, hlp = get_label("slope", "ECG Slope during exercise", "0=Upsloping(Normal), 1=Flat, 2=Downsloping(Bad)")
            slope_opts = ["Upsloping", "Flat", "Downsloping"]
            user_inputs['slope'] = slope_opts.index(st.selectbox(lbl, slope_opts, help=hlp))

            lbl, hlp = get_label("thal", "Blood Flow (Thalassemia test)", "0=Normal, 1=Fixed Defect (Past damage), 2=Reversible (Current block)")
            thal_opts = ["Normal", "Fixed Defect", "Reversible Defect"]
            user_inputs['thal'] = thal_opts.index(st.selectbox(lbl, thal_opts, help=hlp))

    st.markdown("<br>", unsafe_allow_html=True)

    # --- 6. PREDICTION & XAI ---
    if st.button("🔍 Analyze Risk Profile", type="primary"):
        input_df = pd.DataFrame([user_inputs])
        # Ensure columns match training data exactly
        input_df = input_df[feature_names] 
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        st.header("📊 Assessment Results")
        
        # UI Metrics Display
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if prediction == 1:
                st.error("⚠️ **HIGH RISK DETECTED**")
                st.metric(label="Disease Probability", value=f"{probability*100:.1f}%", delta="Requires Attention", delta_color="inverse")
            else:
                st.success("✅ **LOW RISK / NORMAL**")
                st.metric(label="Health Probability", value=f"{(1-probability)*100:.1f}%", delta="Normal Range")
                
        # Explainable AI (SHAP)
        with res_col2:
            st.markdown("### 🧠 AI Rationale")
            st.info("The AI analyzed your data. Here is exactly **why** it made this prediction:")
            
            # Generate SHAP values (Using the modern API)
            explainer = shap.TreeExplainer(model)
            
            # Calling the explainer directly returns an Explanation object
            shap_obj = explainer(input_df) 
            
            # Random Forest returns a 3D array: (samples, features, classes)
            # We want: Patient 0, All features (:), Class 1 (High Risk)
            if len(shap_obj.shape) == 3:
                explanation = shap_obj[0, :, 1]
            else:
                # Fallback just in case the model format is slightly different
                explanation = shap_obj[0] 
            
            # Plotting the SHAP Waterfall via Matplotlib
            fig, ax = plt.subplots(figsize=(6, 4))
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig)
            
            with st.expander("How to read this chart?"):
                st.write("""
                * **Red arrows (Right):** These inputs pushed your risk score **UP**.
                * **Blue arrows (Left):** These inputs pushed your risk score **DOWN** (Healthy).
                * The longer the arrow, the bigger the impact that specific medical test had on your final result.
                """)