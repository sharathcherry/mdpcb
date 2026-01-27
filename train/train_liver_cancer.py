"""
Liver Cancer Prediction Model Training Script
Dataset: Synthetic Liver Cancer Dataset from Kaggle
Features: 14 (13 predictors + 1 target)
Patients: 5,000
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# Load the dataset
print("Loading Liver Cancer dataset...")
df = pd.read_csv(r'c:\Users\katuk\Downloads\synthetic_liver_cancer_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Define target and features
target_column = 'liver_cancer'
feature_columns = [col for col in df.columns if col != target_column]

print(f"\nTarget column: {target_column}")
print(f"Feature columns ({len(feature_columns)}): {feature_columns}")
print(f"\nTarget distribution:")
print(df[target_column].value_counts())
print(f"Positive rate: {df[target_column].mean()*100:.2f}%")

# Identify categorical columns
categorical_cols = df[feature_columns].select_dtypes(include=['object']).columns.tolist()
numerical_cols = df[feature_columns].select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Encode categorical variables
encoders = {}
df_encoded = df.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le
    print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Prepare features and target
X = df_encoded[feature_columns]
y = df_encoded[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Train Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handle class imbalance
)

model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"MODEL PERFORMANCE")
print(f"{'='*50}")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Cancer', 'Cancer']))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance
print(f"\nTop 10 Feature Importances:")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']*100:.2f}%")

# Create packaged model
packaged_model = {
    'model': model,
    'scaler': scaler,
    'encoders': encoders,
    'feature_columns': feature_columns,
    'numerical_columns': numerical_cols,
    'categorical_columns': categorical_cols,
    'accuracy': accuracy,
    'feature_importance': feature_importance.to_dict('records')
}

# Save the model
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'new', 'models')
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, 'liver_cancer_model.sav')

with open(model_path, 'wb') as f:
    pickle.dump(packaged_model, f)
print(f"\n{'='*50}")
print(f"Model saved to: {model_path}")
print(f"{'='*50}")

# Verify saved model
print("\nVerifying saved model...")
with open(model_path, 'rb') as f:
    loaded = pickle.load(f)
test_pred = loaded['model'].predict(X_test_scaled)
loaded_accuracy = accuracy_score(y_test, test_pred)
print(f"Loaded model accuracy: {loaded_accuracy*100:.2f}%")
print("Model verification successful!")

print(f"""
{'='*50}
LIVER CANCER MODEL SUMMARY
{'='*50}
- Dataset: 5,000 patients
- Features: 13 predictors
- Model: Random Forest Classifier
- Accuracy: {accuracy*100:.2f}%
- Positive Rate: {df[target_column].mean()*100:.2f}%

Categorical Encodings:
""")
for col, encoder in encoders.items():
    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print(f"  {col}: {mapping}")
