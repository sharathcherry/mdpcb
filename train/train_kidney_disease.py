"""
Chronic Kidney Disease Prediction Model Training
Dataset: Rabie El Kharoua's CKD Dataset (1,659 patients, 54 features)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import joblib
import os

# Load dataset
print("Loading Chronic Kidney Disease dataset...")
df = pd.read_csv('chronic_kidney_disease.csv')
print(f"Dataset shape: {df.shape}")

# Check class distribution
print(f"\nTarget distribution:")
print(df['Diagnosis'].value_counts())
print(f"Class imbalance ratio: {df['Diagnosis'].value_counts()[1] / df['Diagnosis'].value_counts()[0]:.2f}:1")

# Drop non-predictive columns
columns_to_drop = ['PatientID', 'DoctorInCharge']
df = df.drop(columns=columns_to_drop)
print(f"\nAfter dropping non-predictive columns: {df.shape}")

# All features are already numeric (no encoding needed!)
print("\nAll features are already numeric - no encoding required!")

# Define features and target
feature_columns = [col for col in df.columns if col != 'Diagnosis']
X = df[feature_columns]
y = df['Diagnosis']

print(f"\nFeature columns ({len(feature_columns)}):")
print(feature_columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model with class_weight='balanced' to handle imbalanced data
print("\nTraining Random Forest model with balanced class weights...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # Important for imbalanced data
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"\n{'='*50}")
print(f"MODEL PERFORMANCE")
print(f"{'='*50}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"ROC-AUC Score: {roc_auc:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No CKD', 'CKD']))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Save model package
model_package = {
    'model': model,
    'scaler': scaler,
    'feature_columns': feature_columns,
    'model_type': 'kidney_disease',
    'accuracy': accuracy,
    'roc_auc': roc_auc
}

# Save to models folder
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'new', 'models')
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'kidney_disease_model.sav')
joblib.dump(model_package, model_path)
print(f"\nModel saved to: {model_path}")

# Test prediction
print(f"\n{'='*50}")
print("SAMPLE PREDICTION TEST")
print(f"{'='*50}")

# Create a sample input (healthy person)
sample_healthy = {
    'Age': 45,
    'Gender': 1,  # Female
    'Ethnicity': 0,  # Caucasian
    'SocioeconomicStatus': 1,  # Middle
    'EducationLevel': 2,  # Bachelor's
    'BMI': 24.5,
    'Smoking': 0,  # No
    'AlcoholConsumption': 2.0,
    'PhysicalActivity': 5.0,
    'DietQuality': 7.0,
    'SleepQuality': 7.5,
    'FamilyHistoryKidneyDisease': 0,
    'FamilyHistoryHypertension': 0,
    'FamilyHistoryDiabetes': 0,
    'PreviousAcuteKidneyInjury': 0,
    'UrinaryTractInfections': 0,
    'SystolicBP': 120,
    'DiastolicBP': 80,
    'FastingBloodSugar': 95.0,
    'HbA1c': 5.5,
    'SerumCreatinine': 0.9,  # Normal: 0.7-1.3
    'BUNLevels': 15.0,  # Normal: 7-20
    'GFR': 95.0,  # Normal: >90
    'ProteinInUrine': 0.1,
    'ACR': 10.0,  # Normal: <30
    'SerumElectrolytesSodium': 140.0,
    'SerumElectrolytesPotassium': 4.0,
    'SerumElectrolytesCalcium': 9.5,
    'SerumElectrolytesPhosphorus': 3.5,
    'HemoglobinLevels': 14.0,
    'CholesterolTotal': 180.0,
    'CholesterolLDL': 100.0,
    'CholesterolHDL': 55.0,
    'CholesterolTriglycerides': 120.0,
    'ACEInhibitors': 0,
    'Diuretics': 0,
    'NSAIDsUse': 1.0,
    'Statins': 0,
    'AntidiabeticMedications': 0,
    'Edema': 0,
    'FatigueLevels': 2.0,
    'NauseaVomiting': 0.0,
    'MuscleCramps': 0.5,
    'Itching': 1.0,
    'QualityOfLifeScore': 85.0,
    'HeavyMetalsExposure': 0,
    'OccupationalExposureChemicals': 0,
    'WaterQuality': 0,  # Good
    'MedicalCheckupsFrequency': 2.0,
    'MedicationAdherence': 8.0,
    'HealthLiteracy': 7.0
}

sample_df = pd.DataFrame([sample_healthy])
sample_scaled = scaler.transform(sample_df[feature_columns])
healthy_pred = model.predict(sample_scaled)[0]
healthy_prob = model.predict_proba(sample_scaled)[0][1]
print(f"Healthy person prediction: {'CKD' if healthy_pred == 1 else 'No CKD'} (Risk: {healthy_prob*100:.1f}%)")

# High-risk person
sample_high_risk = sample_healthy.copy()
sample_high_risk['Age'] = 65
sample_high_risk['BMI'] = 32.0
sample_high_risk['Smoking'] = 1
sample_high_risk['FamilyHistoryKidneyDisease'] = 1
sample_high_risk['FamilyHistoryHypertension'] = 1
sample_high_risk['FamilyHistoryDiabetes'] = 1
sample_high_risk['PreviousAcuteKidneyInjury'] = 1
sample_high_risk['SystolicBP'] = 160
sample_high_risk['DiastolicBP'] = 100
sample_high_risk['HbA1c'] = 8.5
sample_high_risk['SerumCreatinine'] = 3.5  # Elevated
sample_high_risk['BUNLevels'] = 40.0  # Elevated
sample_high_risk['GFR'] = 35.0  # Low (Stage 3b CKD)
sample_high_risk['ProteinInUrine'] = 3.5  # High
sample_high_risk['ACR'] = 200.0  # Very high
sample_high_risk['Edema'] = 1
sample_high_risk['FatigueLevels'] = 7.0
sample_high_risk['QualityOfLifeScore'] = 45.0

sample_df = pd.DataFrame([sample_high_risk])
sample_scaled = scaler.transform(sample_df[feature_columns])
risk_pred = model.predict(sample_scaled)[0]
risk_prob = model.predict_proba(sample_scaled)[0][1]
print(f"High-risk person prediction: {'CKD' if risk_pred == 1 else 'No CKD'} (Risk: {risk_prob*100:.1f}%)")

print(f"\n{'='*50}")
print("TRAINING COMPLETE!")
print(f"{'='*50}")
