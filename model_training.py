import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes
import pickle
import os

# 1. Heart Disease (UCI)
heart = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
                    names=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target'])
heart = heart.replace('?', np.nan)
heart = heart.dropna()
heart['target'] = heart['target'].apply(lambda x: 1 if x > 0 else 0)

X_heart = heart.drop('target', axis=1)
y_heart = heart['target']

scaler_heart = StandardScaler()
X_heart_scaled = scaler_heart.fit_transform(X_heart)

heart_model = RandomForestClassifier(n_estimators=100, random_state=42)
heart_model.fit(X_heart_scaled, y_heart)

with open('heart_disease_model.pkl', 'wb') as f:
    pickle.dump(heart_model, f)
with open('scaler_heart.pkl', 'wb') as f:
    pickle.dump(scaler_heart, f)

# 2. Diabetes (sklearn built-in)
diabetes = load_diabetes()
X_diab = diabetes.data
y_diab = (diabetes.target > diabetes.target.mean()).astype(int)  # binary classification

diab_model = RandomForestClassifier(n_estimators=100, random_state=42)
diab_model.fit(X_diab, y_diab)

with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(diab_model, f)

# 3. Breast Cancer (sklearn built-in)
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target  # 0=malignant, 1=benign

cancer_model = RandomForestClassifier(n_estimators=200, random_state=42)
cancer_model.fit(X_cancer, y_cancer)

with open('breast_cancer_model.pkl', 'wb') as f:
    pickle.dump(cancer_model, f)

print("All models trained and saved!")