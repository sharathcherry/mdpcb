"""
Epilepsy Seizure Prediction Model Training
Dataset: Archive/csv/epilepsy.csv
Target: y (1=Seizure, 2-5=Non-seizure) -> Binary conversion
Features: EEG signal measurements (178 features)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# Load data
print("Loading Epilepsy dataset...")
df = pd.read_csv('Archive/csv/epilepsy.csv')
print(f"Dataset shape: {df.shape}")
print(f"Null values: {df.isnull().sum().sum()}")

# Drop identifier column
if 'Unnamed' in df.columns[0]:
    df = df.drop(columns=[df.columns[0]])
    print("Dropped identifier column")

# Original target (1=seizure, 2-5=non-seizure types)
print(f"\nOriginal target distribution:")
print(df['y'].value_counts().sort_index())

# Convert to binary: 1 = Seizure, 0 = No Seizure
# Class 1 = seizure activity
# Classes 2-5 = non-seizure (tumor area, healthy area, eyes closed/open)
df['seizure'] = (df['y'] == 1).astype(int)
print(f"\nBinary target distribution:")
print(df['seizure'].value_counts())
print(f"Class balance: {df['seizure'].value_counts(normalize=True).to_dict()}")

# Separate features and target
target_col = 'seizure'
X = df.drop(columns=['y', target_col])  # Drop original 'y' and new target
y = df[target_col]

# Feature columns (X1 to X178 - EEG measurements)
feature_columns = list(X.columns)
print(f"\nFeatures: {len(feature_columns)} EEG measurements")

# All columns are numerical
numerical_columns = feature_columns
categorical_columns = []

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain set: {len(X_train)}, Test set: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model with balanced weights (20% seizure class)
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # Handle 20% minority class
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
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Seizure', 'Seizure']))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"TN={cm[0,0]}, FP={cm[0,1]}")
print(f"FN={cm[1,0]}, TP={cm[1,1]}")

# Feature importance (top 10)
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 Features by Importance:")
for _, row in importance_df.head(10).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f} ({row['Importance']*100:.1f}%)")

# Save model package
model_package = {
    'model': model,
    'scaler': scaler,
    'encoders': {},  # No encoding needed, all numeric
    'feature_columns': feature_columns,
    'numerical_columns': numerical_columns,
    'categorical_columns': categorical_columns,
    'accuracy': accuracy,
    'feature_importance': feature_importance
}

# Save to models directory
os.makedirs('new/models', exist_ok=True)
model_path = 'new/models/epilepsy_model.sav'
with open(model_path, 'wb') as f:
    pickle.dump(model_package, f)

print(f"\n{'='*60}")
print(f"Model saved to: {model_path}")
print(f"{'='*60}")
