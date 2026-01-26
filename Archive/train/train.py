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
print(f"\nActual columns in dataset:")
for i, col in enumerate(df.columns):
    print(f"{i}: '{col}'")
print()

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

# Check for missing values
print("Checking for missing values...")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
    df = df.dropna()
    print(f"Dropped rows with missing values. New shape: {df.shape}\n")
else:
    print("No missing values found.\n")

# Encode categorical variables
print("Encoding categorical variables...")
label_encoders = {}

# Get all categorical columns
categorical_columns = []
for col in df.columns:
    if col != target_col and col != 'Country' and df[col].dtype == 'object':
        categorical_columns.append(col)

print(f"Categorical columns to encode: {categorical_columns}\n")

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded: {col} - Classes: {len(le.classes_)}")

# Encode target variable
le_target = LabelEncoder()
y = le_target.fit_transform(df[target_col])
print(f"\nTarget classes: {le_target.classes_}")

# Prepare features
feature_columns = [col for col in df.columns if col not in [target_col, 'Country']]
X = df[feature_columns]

print(f"\nFeature columns ({len(feature_columns)}):")
for i, col in enumerate(feature_columns):
    print(f"  {i}: {col}")

print(f"\nTarget distribution:")
print(pd.Series(y).value_counts())
print()

# Split the data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}\n")

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

# Evaluate the model
print("Evaluating model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
print("\nTop 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))

# Create models directory
os.makedirs('models', exist_ok=True)

# Save the trained model
print("\nSaving model...")
with open('models/alzheimers_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to 'models/alzheimers_model.pkl'")

# Save the label encoders
with open('models/alzheimers_label_encoders.pkl', 'wb') as f:
    pickle.dump({
        'categorical_encoders': label_encoders,
        'target_encoder': le_target,
        'feature_columns': feature_columns,
        'target_column': target_col
    }, f)

print("Label encoders saved to 'models/alzheimers_label_encoders.pkl'")

# Save feature names with UTF-8 encoding
with open('models/alzheimers_features.txt', 'w', encoding='utf-8') as f:  # Added encoding='utf-8'
    f.write(f"Target column: {target_col}\n\n")
    f.write("Feature columns in order:\n")
    for i, col in enumerate(feature_columns):
        f.write(f"{i}: {col}\n")

print("\nTraining complete! Model ready for use.")