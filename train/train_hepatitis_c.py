"""
Hepatitis C Prediction Model Training Script
Dataset: Hepatitis C Prediction Dataset from Kaggle/UCI ML Repository
Features: 14 (12 predictors + 1 target + 1 ID)
Patients: 615 (Blood donors + Hepatitis C patients with disease stages)
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
print("Loading Hepatitis C dataset...")
df = pd.read_csv(r'c:\Users\katuk\Downloads\HepatitisCdata.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Drop ID column
df = df.drop('Unnamed: 0', axis=1)

# Check target distribution
print(f"\nTarget distribution (Category):")
print(df['Category'].value_counts())

# Handle missing values - impute with median for numerical columns
print(f"\nMissing values before imputation:")
print(df.isnull().sum())

numerical_cols = ['ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print(f"\nMissing values after imputation:")
print(df.isnull().sum().sum())

# Create binary classification: Healthy (Blood Donor) vs Disease (any Hepatitis stage)
# Also keep multi-class for disease staging
df['Binary_Target'] = df['Category'].apply(lambda x: 0 if x in ['0=Blood Donor', '0s=suspect Blood Donor'] else 1)

print(f"\nBinary Target Distribution:")
print(df['Binary_Target'].value_counts())
print(f"Disease rate: {df['Binary_Target'].mean()*100:.2f}%")

# Define features
target_column = 'Binary_Target'
exclude_cols = ['Category', 'Binary_Target']
feature_columns = [col for col in df.columns if col not in exclude_cols]

print(f"\nFeature columns ({len(feature_columns)}): {feature_columns}")

# Encode categorical variables
encoders = {}

# Encode Sex
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])
encoders['Sex'] = le_sex
print(f"Encoded Sex: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")

# Also encode the multi-class Category for future use
le_category = LabelEncoder()
df['Category_Encoded'] = le_category.fit_transform(df['Category'])
encoders['Category'] = le_category
print(f"Encoded Category: {dict(zip(le_category.classes_, le_category.transform(le_category.classes_)))}")

# Prepare features and target
X = df[feature_columns]
y = df[target_column]

# Identify numerical columns for scaling (all except Sex which is already encoded)
numerical_features = [col for col in feature_columns if col != 'Sex']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

# Train Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
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
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Hepatitis C']))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance
print(f"\nFeature Importances:")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']*100:.2f}%")

# Create packaged model
packaged_model = {
    'model': model,
    'scaler': scaler,
    'encoders': encoders,
    'feature_columns': feature_columns,
    'numerical_columns': numerical_features,
    'categorical_columns': ['Sex'],
    'accuracy': accuracy,
    'feature_importance': feature_importance.to_dict('records'),
    'category_mapping': dict(zip(le_category.classes_, le_category.transform(le_category.classes_)))
}

# Save the model
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'new', 'models')
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, 'hepatitis_c_model.sav')

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
HEPATITIS C MODEL SUMMARY
{'='*50}
- Dataset: 615 patients (blood donors + hepatitis patients)
- Features: {len(feature_columns)} predictors
- Model: Random Forest Classifier
- Accuracy: {accuracy*100:.2f}%
- Disease Rate: {df['Binary_Target'].mean()*100:.2f}%

Laboratory Values Used:
- ALB: Albumin
- ALP: Alkaline Phosphatase  
- ALT: Alanine Aminotransferase
- AST: Aspartate Aminotransferase
- BIL: Bilirubin
- CHE: Cholinesterase
- CHOL: Cholesterol
- CREA: Creatinine
- GGT: Gamma-Glutamyl Transferase
- PROT: Total Protein

Sex Encoding: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}
""")
