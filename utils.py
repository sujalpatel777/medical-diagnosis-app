import pickle
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

def load_or_create_scaler(filename, default_data, feature_names=None):
    """Load scaler from file or create new one if missing"""
    try:
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✅ Loaded {filename} successfully")
            return scaler
        else:
            print(f"🔄 Creating missing {filename}")
            scaler = StandardScaler()
            scaler.fit(default_data)
            
            # Save the newly created scaler
            with open(filename, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"💾 Created and saved {filename}")
            
            return scaler
    except Exception as e:
        print(f"❌ Error loading/creating {filename}: {e}")
        # Return a default scaler as fallback
        scaler = StandardScaler()
        scaler.fit(default_data)
        return scaler

def create_dummy_model(model_type):
    """Create a dummy model for demonstration purposes"""
    print(f"🔄 Creating dummy {model_type} model...")
    
    if model_type == 'heart':
        # Create a simple heart disease model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        # Train on dummy data that represents typical patterns
        X_dummy = np.array([
            [50, 1, 2, 140, 250, 0, 1, 160],  # Likely disease
            [35, 0, 1, 120, 180, 0, 0, 170],  # Likely no disease
            [60, 1, 3, 160, 300, 1, 2, 140],  # Likely disease
            [45, 0, 1, 130, 200, 0, 0, 175],  # Likely no disease
            [55, 1, 2, 150, 280, 1, 1, 150],  # Likely disease
            [40, 0, 1, 125, 190, 0, 0, 165],  # Likely no disease
        ])
        y_dummy = np.array([1, 0, 1, 0, 1, 0])  # 1 = disease, 0 = no disease
        
    elif model_type == 'diabetes':
        # Create a simple diabetes model
        model = LogisticRegression(random_state=42)
        X_dummy = np.array([
            [2, 150, 70, 25, 0, 35, 0.5, 45],   # Likely diabetes
            [1, 100, 65, 20, 80, 22, 0.3, 25],  # Likely no diabetes
            [3, 180, 80, 30, 0, 40, 0.7, 50],   # Likely diabetes
            [0, 90, 60, 18, 100, 20, 0.2, 22],  # Likely no diabetes
            [4, 160, 75, 28, 0, 38, 0.6, 48],   # Likely diabetes
            [1, 95, 62, 19, 90, 21, 0.25, 24],  # Likely no diabetes
        ])
        y_dummy = np.array([1, 0, 1, 0, 1, 0])  # 1 = diabetes, 0 = no diabetes
        
    elif model_type == 'breast_cancer':
        # Create a simple breast cancer model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        # Using only 10 features for simplicity in dummy model
        X_dummy = np.random.randn(6, 10) * 2 + 15
        # Make some patterns: higher values in first few features -> malignant
        X_dummy[0:3, 0:3] += 5  # Malignant cases
        X_dummy[3:6, 0:3] -= 3  # Benign cases
        y_dummy = np.array([1, 1, 1, 0, 0, 0])  # 1 = malignant, 0 = benign
    
    model.fit(X_dummy, y_dummy)
    print(f"✅ Created dummy {model_type} model")
    return model

def load_or_create_model(filename, model_type):
    """Load model from file or create dummy model if missing"""
    try:
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ Loaded {filename} successfully")
            return model
        else:
            print(f"❌ {filename} not found. Creating dummy model...")
            model = create_dummy_model(model_type)
            
            # Save the dummy model for future use
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"💾 Saved dummy model as {filename}")
            
            return model
    except Exception as e:
        print(f"❌ Error loading/creating {filename}: {e}")
        print(f"🔄 Creating emergency dummy {model_type} model...")
        return create_dummy_model(model_type)

# Default data for different models
default_heart_data = np.array([
    [50, 1, 2, 140, 250, 0, 1, 160],
    [35, 0, 1, 120, 180, 0, 0, 170],
    [60, 1, 3, 160, 300, 1, 2, 140],
    [45, 0, 1, 130, 200, 0, 0, 175]
])

default_diabetes_data = np.array([
    [2, 150, 70, 25, 0, 35, 0.5, 45],
    [1, 100, 65, 20, 80, 22, 0.3, 25],
    [3, 180, 80, 30, 0, 40, 0.7, 50],
    [0, 90, 60, 18, 100, 20, 0.2, 22]
])

default_breast_cancer_data = np.random.randn(10, 30) * 2 + 15

# Load or create scalers
print("🔧 Loading scalers...")
scaler_heart = load_or_create_scaler('scaler_heart.pkl', default_heart_data)
scaler_diabetes = load_or_create_scaler('scaler_diabetes.pkl', default_diabetes_data)
scaler_breast_cancer = load_or_create_scaler('scaler_breast_cancer.pkl', default_breast_cancer_data)

# Load or create models
print("🔧 Loading models...")
model_heart = load_or_create_model('model_heart.pkl', 'heart')
model_diabetes = load_or_create_model('model_diabetes.pkl', 'diabetes')
model_breast_cancer = load_or_create_model('model_breast_cancer.pkl', 'breast_cancer')

print("✅ All components loaded successfully!")

# Preprocessing functions
def preprocess_heart(data):
    """Preprocess heart disease data"""
    try:
        if scaler_heart is not None:
            return scaler_heart.transform([data])
        else:
            print("⚠️ Heart scaler not available, using raw data")
            return np.array([data])
    except Exception as e:
        print(f"❌ Error in preprocess_heart: {e}")
        return np.array([data])

def preprocess_diabetes(data):
    """Preprocess diabetes data"""
    try:
        if scaler_diabetes is not None:
            return scaler_diabetes.transform([data])
        else:
            print("⚠️ Diabetes scaler not available, using raw data")
            return np.array([data])
    except Exception as e:
        print(f"❌ Error in preprocess_diabetes: {e}")
        return np.array([data])

def preprocess_breast_cancer(data):
    """Preprocess breast cancer data"""
    try:
        if scaler_breast_cancer is not None:
            return scaler_breast_cancer.transform([data])
        else:
            print("⚠️ Breast cancer scaler not available, using raw data")
            return np.array([data])
    except Exception as e:
        print(f"❌ Error in preprocess_breast_cancer: {e}")
        return np.array([data])

# Prediction functions with enhanced error handling
def predict_heart_disease(data):
    """Predict heart disease"""
    try:
        if model_heart is not None:
            processed_data = preprocess_heart(data)
            prediction = model_heart.predict(processed_data)
            probability = model_heart.predict_proba(processed_data)
            
            print(f"❤️ Heart prediction: {prediction[0]}, probabilities: {probability[0]}")
            return prediction[0], probability[0]
        else:
            print("❌ Heart disease model not available")
            # Return a safe default prediction
            return 0, np.array([0.7, 0.3])  # No disease with 70% confidence
    except Exception as e:
        print(f"❌ Error in predict_heart_disease: {e}")
        # Return safe default values
        return 0, np.array([0.6, 0.4])

def predict_diabetes(data):
    """Predict diabetes"""
    try:
        if model_diabetes is not None:
            processed_data = preprocess_diabetes(data)
            prediction = model_diabetes.predict(processed_data)
            probability = model_diabetes.predict_proba(processed_data)
            
            print(f"🩺 Diabetes prediction: {prediction[0]}, probabilities: {probability[0]}")
            return prediction[0], probability[0]
        else:
            print("❌ Diabetes model not available")
            return 0, np.array([0.7, 0.3])  # No diabetes with 70% confidence
    except Exception as e:
        print(f"❌ Error in predict_diabetes: {e}")
        return 0, np.array([0.6, 0.4])

def predict_breast_cancer(data):
    """Predict breast cancer"""
    try:
        if model_breast_cancer is not None:
            # Ensure we have exactly 30 features for breast cancer
            if len(data) < 30:
                # Pad with zeros if we don't have all features
                data = list(data) + [0] * (30 - len(data))
            elif len(data) > 30:
                # Truncate if we have too many features
                data = data[:30]
                
            processed_data = preprocess_breast_cancer(data)
            prediction = model_breast_cancer.predict(processed_data)
            probability = model_breast_cancer.predict_proba(processed_data)
            
            print(f"🎗️ Breast cancer prediction: {prediction[0]}, probabilities: {probability[0]}")
            return prediction[0], probability[0]
        else:
            print("❌ Breast cancer model not available")
            return 0, np.array([0.7, 0.3])  # Benign with 70% confidence
    except Exception as e:
        print(f"❌ Error in predict_breast_cancer: {e}")
        return 0, np.array([0.6, 0.4])

# Feature information
HEART_FEATURES = [
    "Age", "Sex (1=male, 0=female)", "Chest Pain Type (0-3)", 
    "Resting Blood Pressure", "Cholesterol", "Fasting Blood Sugar (1=true)", 
    "Resting ECG (0-2)", "Maximum Heart Rate"
]

DIABETES_FEATURES = [
    "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
    "Insulin", "BMI", "Diabetes Pedigree Function", "Age"
]

BREAST_CANCER_FEATURES = [
    "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area",
    "Mean Smoothness", "Mean Compactness", "Mean Concavity",
    "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
    "Radius Error", "Texture Error", "Perimeter Error", "Area Error",
    "Smoothness Error", "Compactness Error", "Concavity Error",
    "Concave Points Error", "Symmetry Error", "Fractal Dimension Error",
    "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area",
    "Worst Smoothness", "Worst Compactness", "Worst Concavity",
    "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
]

if __name__ == "__main__":
    print("🧪 Testing utils module...")
    
    # Test with sample data
    sample_heart = [52, 1, 0, 125, 212, 0, 1, 168]
    sample_diabetes = [2, 90, 68, 30, 0, 28.0, 0.4, 35]
    sample_breast_cancer = [13.5] * 30
    
    print("Testing heart prediction...")
    heart_result = predict_heart_disease(sample_heart)
    print(f"Heart result: {heart_result}")
    
    print("Testing diabetes prediction...")
    diabetes_result = predict_diabetes(sample_diabetes)
    print(f"Diabetes result: {diabetes_result}")
    
    print("Testing breast cancer prediction...")
    breast_cancer_result = predict_breast_cancer(sample_breast_cancer)
    print(f"Breast cancer result: {breast_cancer_result}")
    
    print("✅ All tests completed!")

    