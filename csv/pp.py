import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('alzheimers_prediction_dataset.csv')
print(f"Dataset shape: {df.shape}")

# Find the target column
target_col = None
for col in df.columns:
    if 'diagnosis' in col.lower() or 'alzheimer' in col.lower():
        target_col = col
        break

if target_col is None:
    target_col = df.columns[-1]

print(f"Using target column: '{target_col}'")
print(f"Unique values in target: {df[target_col].unique()}\n")

# Handle missing values
df = df.dropna()

# Encode target variable FIRST (before modifying df)
le_target = LabelEncoder()
y = le_target.fit_transform(df[target_col])
print(f"Target classes: {le_target.classes_}")

# Prepare features (exclude target and Country)
feature_columns = [col for col in df.columns if col not in [target_col, 'Country']]
X = df[feature_columns].copy()

# Now encode categorical features in X (not in df)
print("\nEncoding categorical variables...")
label_encoders = {}

for col in feature_columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded: {col} - Classes: {len(le.classes_)}")

print(f"\nFeature columns ({len(feature_columns)}):")
for i, col in enumerate(feature_columns):
    print(f"  {i}: {col}")

print(f"\nTarget distribution:")
print(pd.Series(y).value_counts())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]}, Testing set: {X_test.shape[0]}\n")

# Train the model
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)
print("Model training complete!\n")

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Create models directory
os.makedirs('models', exist_ok=True)

# Save model
with open('models/alzheimers_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved to 'models/alzheimers_model.pkl'")

# Save encoders (ONLY feature encoders, not target in the dict)
with open('models/alzheimers_label_encoders.pkl', 'wb') as f:
    pickle.dump({
        'categorical_encoders': label_encoders,
        'target_encoder': le_target,
        'feature_columns': feature_columns
    }, f)
print("Label encoders saved")

# Save feature info
with open('models/alzheimers_features.txt', 'w', encoding='utf-8') as f:
    f.write(f"Target column: {target_col}\n\n")
    f.write("Feature columns:\n")
    for i, col in enumerate(feature_columns):
        f.write(f"{i}: {col}\n")

print("\nTraining complete!")