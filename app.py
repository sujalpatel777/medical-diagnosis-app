import streamlit as st
import numpy as np
from utils import (
    predict_heart_disease, predict_diabetes, predict_breast_cancer,
    HEART_FEATURES, DIABETES_FEATURES, BREAST_CANCER_FEATURES
)

st.set_page_config(page_title="Medical Diagnosis App", page_icon="🏥", layout="wide")

st.title("🏥 Medical Diagnosis Prediction App")
st.markdown("""
This app predicts the likelihood of various medical conditions based on patient data.
**Note:** This is for educational purposes only. Always consult healthcare professionals for medical diagnoses.
""")

# Sidebar for model selection
st.sidebar.title("Navigation")
model_choice = st.sidebar.selectbox(
    "Choose Prediction Model",
    ["Heart Disease", "Diabetes", "Breast Cancer"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Enter patient data in the main panel and click 'Predict' to get results."
)

def heart_disease_page():
    st.header("❤️ Heart Disease Prediction")
    st.markdown("Enter the patient's information below:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50, key="heart_age")
        sex = st.selectbox("Sex", ["Female", "Male"], key="heart_sex")
        cp = st.selectbox("Chest Pain Type", 
                         ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], 
                         key="heart_cp")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 
                                  min_value=50, max_value=250, value=120, key="heart_trestbps")
    
    with col2:
        chol = st.number_input("Serum Cholesterol (mg/dl)", 
                              min_value=100, max_value=600, value=200, key="heart_chol")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], key="heart_fbs")
        restecg = st.selectbox("Resting Electrocardiographic Results",
                              ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
                              key="heart_restecg")
        thalach = st.number_input("Maximum Heart Rate Achieved", 
                                 min_value=50, max_value=250, value=150, key="heart_thalach")
    
    # Convert inputs to model format
    sex_num = 1 if sex == "Male" else 0
    cp_num = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
    fbs_num = 1 if fbs == "Yes" else 0
    restecg_num = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
    
    input_data = [age, sex_num, cp_num, trestbps, chol, fbs_num, restecg_num, thalach]
    
    if st.button("Predict Heart Disease", type="primary"):
        with st.spinner("Analyzing..."):
            prediction, probability = predict_heart_disease(input_data)
            
        if prediction is not None:
            st.success("Prediction completed!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Prediction",
                    "Disease Detected" if prediction == 1 else "No Disease",
                    delta="Positive" if prediction == 1 else "Negative",
                    delta_color="inverse" if prediction == 1 else "normal"
                )
            
            with col2:
                confidence = probability[1] if prediction == 1 else probability[0]
                st.metric(
                    "Confidence",
                    f"{confidence:.2%}",
                    delta=f"{(confidence-0.5):.2%}",
                    delta_color="normal"
                )
            
            # Probability visualization
            st.subheader("Probability Distribution")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.progress(probability[0], text=f"No Disease: {probability[0]:.2%}")
            with prob_col2:
                st.progress(probability[1], text=f"Disease: {probability[1]:.2%}")
        else:
            st.error("Unable to make prediction. Please check if the model is properly loaded.")

def diabetes_page():
    st.header("🩺 Diabetes Prediction")
    st.markdown("Enter the patient's information below:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1, key="diab_preg")
        glucose = st.number_input("Glucose Level (mg/dl)", min_value=0, max_value=300, value=100, key="diab_glucose")
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70, key="diab_bp")
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, key="diab_skin")
    
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=300, value=80, key="diab_insulin")
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, key="diab_bmi")
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, key="diab_pedigree")
        age = st.number_input("Age", min_value=1, max_value=120, value=30, key="diab_age")
    
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
    
    if st.button("Predict Diabetes", type="primary"):
        with st.spinner("Analyzing..."):
            prediction, probability = predict_diabetes(input_data)
            
        if prediction is not None:
            st.success("Prediction completed!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Prediction",
                    "Diabetes Detected" if prediction == 1 else "No Diabetes",
                    delta="Positive" if prediction == 1 else "Negative",
                    delta_color="inverse" if prediction == 1 else "normal"
                )
            
            with col2:
                confidence = probability[1] if prediction == 1 else probability[0]
                st.metric(
                    "Confidence",
                    f"{confidence:.2%}",
                    delta=f"{(confidence-0.5):.2%}",
                    delta_color="normal"
                )
            
            # Probability visualization
            st.subheader("Probability Distribution")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.progress(probability[0], text=f"No Diabetes: {probability[0]:.2%}")
            with prob_col2:
                st.progress(probability[1], text=f"Diabetes: {probability[1]:.2%}")
        else:
            st.error("Unable to make prediction. Please check if the model is properly loaded.")

def breast_cancer_page():
    st.header("🎗️ Breast Cancer Prediction")
    st.markdown("""
    This model uses 30 features from breast mass characteristics.
    For demonstration, we'll use simplified inputs. In practice, all 30 features should be provided.
    """)
    
    st.warning("Note: This is a simplified interface. A real application would require all 30 medical features.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mean_radius = st.number_input("Mean Radius", min_value=0.0, max_value=30.0, value=13.0, key="bc_radius")
        mean_texture = st.number_input("Mean Texture", min_value=0.0, max_value=40.0, value=20.0, key="bc_texture")
        mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0, max_value=200.0, value=85.0, key="bc_perimeter")
        mean_area = st.number_input("Mean Area", min_value=0.0, max_value=2500.0, value=550.0, key="bc_area")
        mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0, max_value=0.3, value=0.1, key="bc_smoothness")
    
    with col2:
        mean_compactness = st.number_input("Mean Compactness", min_value=0.0, max_value=0.5, value=0.1, key="bc_compactness")
        mean_concavity = st.number_input("Mean Concavity", min_value=0.0, max_value=0.5, value=0.1, key="bc_concavity")
        worst_radius = st.number_input("Worst Radius", min_value=0.0, max_value=40.0, value=15.0, key="bc_worst_radius")
        worst_texture = st.number_input("Worst Texture", min_value=0.0, max_value=50.0, value=25.0, key="bc_worst_texture")
        worst_area = st.number_input("Worst Area", min_value=0.0, max_value=5000.0, value=700.0, key="bc_worst_area")
    
    # Create a simplified feature vector (in practice, you'd need all 30 features)
    input_data = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
                 mean_compactness, mean_concavity] + [0.1] * 23  # Padding with default values
    
    if st.button("Predict Breast Cancer", type="primary"):
        with st.spinner("Analyzing..."):
            prediction, probability = predict_breast_cancer(input_data)
            
        if prediction is not None:
            st.success("Prediction completed!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Prediction",
                    "Malignant" if prediction == 1 else "Benign",
                    delta="Malignant" if prediction == 1 else "Benign",
                    delta_color="inverse" if prediction == 1 else "normal"
                )
            
            with col2:
                confidence = probability[1] if prediction == 1 else probability[0]
                st.metric(
                    "Confidence",
                    f"{confidence:.2%}",
                    delta=f"{(confidence-0.5):.2%}",
                    delta_color="normal"
                )
            
            # Probability visualization
            st.subheader("Probability Distribution")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.progress(probability[0], text=f"Benign: {probability[0]:.2%}")
            with prob_col2:
                st.progress(probability[1], text=f"Malignant: {probability[1]:.2%}")
        else:
            st.error("Unable to make prediction. Please check if the model is properly loaded.")

# Main app logic
if model_choice == "Heart Disease":
    heart_disease_page()
elif model_choice == "Diabetes":
    diabetes_page()
elif model_choice == "Breast Cancer":
    breast_cancer_page()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>⚠️ <strong>Disclaimer:</strong> This application is for educational and demonstration purposes only. 
        It should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice.</p>
    </div>
    """,
    unsafe_allow_html=True
)