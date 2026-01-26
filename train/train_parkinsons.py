"""
Parkinson's Disease Prediction Model Training
Dataset: Rabie El Kharoua's Parkinson's Disease Dataset (2,105 patients, 33 features)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Load the dataset
print("Loading Parkinson's Disease dataset...")
df = pd.read_csv('parkinsons_disease_data.csv')

print(f"Dataset shape: {df.shape}")
print(f"Total patients: {len(df)}")

# Check target distribution
print(f"\nTarget distribution (Diagnosis):")
print(df['Diagnosis'].value_counts())

# Drop non-predictive columns
columns_to_drop = ['PatientID', 'DoctorInCharge']
df = df.drop(columns=columns_to_drop)

# Separate features and target
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Store feature columns
feature_columns = list(X.columns)
print(f"\nFeatures ({len(feature_columns)}): {feature_columns}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"MODEL PERFORMANCE")
print(f"{'='*60}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Parkinson\'s', 'Parkinson\'s']))

# Feature importance
print(f"\nTop 10 Features:")
feat_imp = pd.DataFrame({'feature': feature_columns, 'importance': model.feature_importances_})
feat_imp = feat_imp.sort_values('importance', ascending=False)
for _, row in feat_imp.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']*100:.2f}%")

# Save the model
model_package = {
    'model': model,
    'scaler': scaler,
    'feature_columns': feature_columns,
    'encoders': {},
    'accuracy': accuracy,
    'model_type': 'RandomForestClassifier',
    'n_features': len(feature_columns)
}

models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'new', 'models')
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, 'parkinsons_model.sav')
joblib.dump(model_package, model_path)
print(f"\n✅ Model saved to: {model_path}")

# Test sample predictions
print(f"\n{'='*60}")
print(f"SAMPLE PREDICTIONS")
print(f"{'='*60}")

# Healthy sample
healthy = [55, 1, 0, 2, 24.5, 0, 2.0, 7.0, 8.0, 8.0, 0, 0, 0, 0, 0, 0,
           120, 80, 200, 100, 60, 150, 15, 28, 9.0, 0, 0, 0, 0, 0, 0, 0]
healthy_scaled = scaler.transform([healthy])
pred = model.predict(healthy_scaled)[0]
prob = model.predict_proba(healthy_scaled)[0]
print(f"Healthy Profile: {'Parkinson' if pred == 1 else 'No Parkinson'} (Prob: {prob[1]*100:.1f}%)")

# High-risk sample
risk = [75, 0, 0, 1, 28.0, 1, 10.0, 2.0, 4.0, 5.0, 1, 1, 1, 1, 1, 0,
        160, 95, 280, 180, 35, 350, 150, 18, 3.0, 1, 1, 1, 1, 1, 1, 1]
risk_scaled = scaler.transform([risk])
pred = model.predict(risk_scaled)[0]
prob = model.predict_proba(risk_scaled)[0]
print(f"High-Risk Profile: {'Parkinson' if pred == 1 else 'No Parkinson'} (Prob: {prob[1]*100:.1f}%)")

print(f"\n{'='*60}")
print(f"TRAINING COMPLETE!")
print(f"{'='*60}")
