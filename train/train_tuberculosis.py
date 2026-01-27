"""
Train Tuberculosis (TB) Risk Prediction Model
Dataset: tuberculosis_model.csv
Target: TB Positive/Negative based on symptoms
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TUBERCULOSIS (TB) RISK PREDICTION MODEL TRAINING")
print("="*70)

# Load data
df = pd.read_csv('Archive/csv/tuberculosis_model.csv')
print(f"\nDataset shape: {df.shape}")

# Drop ID columns
df = df.drop(['no', 'name'], axis=1)

# Encode gender
gender_encoder = LabelEncoder()
df['gender'] = gender_encoder.fit_transform(df['gender'])

# Get symptom columns
symptom_cols = [c for c in df.columns if c != 'gender']
print(f"\nSymptom columns ({len(symptom_cols)}): {symptom_cols}")

# Feature engineering - create aggregate symptom scores
df['respiratory_symptoms'] = (df['coughing blood'] + df['sputum mixed with blood'] + 
                              df['shortness of breath'] + 
                              df['cough and phlegm continuously for two weeks to four weeks'])

df['systemic_symptoms'] = (df['fever for two weeks'] + df['night sweats '] + 
                           df['weight loss '] + df['body feels tired'])

df['lymph_symptoms'] = (df['lumps that appear around the armpits and neck'] + 
                        df['swollen lymph nodes'])

df['symptom_count'] = df[symptom_cols].sum(axis=1)

# Create target: 7+ symptoms = TB Positive (high risk)
df['tb_positive'] = (df['symptom_count'] >= 7).astype(int)

print(f"\nTarget distribution:")
print(df['tb_positive'].value_counts())
print(f"0 = TB Negative/Low Risk, 1 = TB Positive/High Risk")

# Features and target
feature_columns = [c for c in df.columns if c not in ['tb_positive']]
X = df[feature_columns]
y = df['tb_positive']

print(f"\nFeatures ({len(feature_columns)}): {feature_columns}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
print("\n" + "="*70)
print("Training Extra Trees Classifier with Feature Engineering...")
print("="*70)

model = ExtraTreesClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)

# Cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"CV Mean Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# Fit model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Set Accuracy: {accuracy*100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['TB Negative', 'TB Positive']))

# Feature importance
feature_importance = dict(zip(feature_columns, model.feature_importances_))
print("\nTop Feature Importances:")
for feat, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {feat}: {imp:.4f}")

# Save model
model_package = {
    'model': model,
    'scaler': scaler,
    'gender_encoder': gender_encoder,
    'feature_columns': feature_columns,
    'symptom_columns': symptom_cols,
    'numerical_columns': feature_columns,
    'categorical_columns': [],
    'encoders': {'gender': gender_encoder},
    'accuracy': accuracy,
    'cv_accuracy': cv_scores.mean(),
    'model_type': 'Extra Trees + Feature Engineering',
    'feature_importance': feature_importance,
    'classes': ['TB Negative', 'TB Positive']
}

with open('new/models/tuberculosis_model.sav', 'wb') as f:
    pickle.dump(model_package, f)

print("\n" + "="*70)
print(f"Model saved to: new/models/tuberculosis_model.sav")
print(f"Final Test Accuracy: {accuracy*100:.2f}%")
print(f"CV Accuracy: {cv_scores.mean()*100:.2f}%")
print("="*70)
