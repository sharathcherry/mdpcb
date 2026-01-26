import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Load dataset
print("Loading COPD dataset...")
df = pd.read_csv('finalalldata.csv')
print(f"Dataset shape: {df.shape}\n")
print("Columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Check data info
print("\n" + "="*60)
print("DATASET INFORMATION")
print("="*60)
print(f"\nData types:\n{df.dtypes}\n")
print(f"Missing values:\n{df.isnull().sum()}\n")

# Check unique values in target columns
print(f"Unique values in 'label': {sorted(df['label'].unique())}")
print(f"Label distribution:\n{df['label'].value_counts()}\n")

print(f"Unique values in 'class': {df['class'].unique()}")
print(f"Class distribution:\n{df['class'].value_counts()}\n")

# Drop uid column (identifier, not a feature)
df = df.drop(columns=['uid'], errors='ignore')

# Handle missing values
print(f"Missing values before handling:\n{df.isnull().sum()}\n")
df = df.dropna()
print(f"After dropping nulls: {df.shape}\n")

# Decide target variable - using 'label' (appears to be numeric 0/1)
target_col = 'label'
y = df[target_col].values

# Drop target and redundant columns
exclude_cols = [target_col, 'class']  # 'class' seems to be same as 'label' but text format
feature_columns = [col for col in df.columns if col not in exclude_cols]
X = df[feature_columns].copy()

print(f"\n{'='*60}")
print("FEATURES AND TARGET")
print(f"{'='*60}")
print(f"Target column: {target_col}")
print(f"Number of features: {len(feature_columns)}")
print(f"Features: {feature_columns}\n")

# Check target distribution
unique_targets = sorted(np.unique(y))
print(f"Target classes: {unique_targets}")
print(f"Target distribution:")
for target in unique_targets:
    count = sum(y == target)
    percentage = count / len(y) * 100
    print(f"  Class {target}: {count} samples ({percentage:.1f}%)")
print()

# Check if binary classification
if len(unique_targets) == 2:
    class_names = ['No COPD', 'COPD']
    print("Classification Type: Binary (COPD vs No COPD)\n")
else:
    class_names = [f'Class {i}' for i in unique_targets]
    print(f"Classification Type: Multi-class ({len(unique_targets)} classes)\n")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples\n")

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("="*60)
print("TRAINING RANDOM FOREST CLASSIFIER")
print("="*60)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced',  # Handle class imbalance
    n_jobs=-1,
    verbose=1
)

model.fit(X_train_scaled, y_train)
print("\nTraining complete!\n")

# Evaluate
print("="*60)
print("MODEL EVALUATION")
print("="*60)

y_pred = model.predict(X_test_scaled)

# Get probabilities
y_pred_proba = model.predict_proba(X_test_scaled)
if y_pred_proba.shape[1] == 2:
    y_pred_proba_positive = y_pred_proba[:, 1]
else:
    y_pred_proba_positive = None

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# ROC-AUC (for binary classification)
if len(unique_targets) == 2 and y_pred_proba_positive is not None:
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba_positive)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    except Exception as e:
        print(f"ROC-AUC: Could not calculate - {str(e)}")
        roc_auc = None
else:
    roc_auc = None

# Classification Report
print(f"\nClassification Report:")
labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
target_names = class_names if len(class_names) == len(labels) else [f'Class {i}' for i in labels]
print(classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0))

# Confusion Matrix
print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=labels)
print(cm)
if len(labels) == 2:
    print(f"\nTrue Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")
else:
    print(f"\nRows = Actual, Columns = Predicted")

# Feature importance
feature_imp = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n{'='*60}")
print("FEATURE IMPORTANCE (Top 10)")
print(f"{'='*60}")
print(feature_imp.head(10).to_string(index=False))

# Cross-validation
from sklearn.model_selection import cross_val_score
print(f"\n{'='*60}")
print("CROSS-VALIDATION (5-fold)")
print(f"{'='*60}")
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"Individual fold scores: {[f'{score:.4f}' for score in cv_scores]}")

# Create models directory
os.makedirs('models', exist_ok=True)

# Save model
print(f"\n{'='*60}")
print("SAVING MODEL FILES")
print(f"{'='*60}")

with open('models/copd_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved to 'models/copd_model.pkl'")

# Save scaler
with open('models/copd_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved to 'models/copd_scaler.pkl'")

# Save metadata
metadata = {
    'feature_columns': feature_columns,
    'target_column': target_col,
    'accuracy': accuracy,
    'roc_auc': roc_auc,
    'classes': list(labels) if not isinstance(labels, list) else labels,
    'class_names': class_names,
    'model_type': 'binary' if len(unique_targets) == 2 else 'multiclass',
    'n_features': len(feature_columns),
    'feature_names': feature_columns
}

with open('models/copd_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("✓ Metadata saved to 'models/copd_metadata.pkl'")

# Save feature info as text file
with open('models/copd_features.txt', 'w', encoding='utf-8') as f:
    f.write(f"COPD Prediction Model\n")
    f.write(f"{'='*60}\n\n")
    f.write(f"Model Type: {'Binary Classification' if len(unique_targets) == 2 else 'Multi-class Classification'}\n")
    f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    if roc_auc:
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
    f.write(f"Training samples: {X_train.shape[0]}\n")
    f.write(f"Testing samples: {X_test.shape[0]}\n\n")
    f.write(f"Target column: {target_col}\n")
    f.write(f"Classes: {list(labels) if not isinstance(labels, list) else labels}\n\n")
    f.write(f"Feature columns ({len(feature_columns)}):\n")
    for i, col in enumerate(feature_columns):
        importance = feature_imp[feature_imp['feature'] == col]['importance'].values[0]
        f.write(f"  {i+1}. {col} (importance: {importance:.4f})\n")
    f.write(f"\n{'='*60}\n")
    f.write("Feature Importance Ranking:\n")
    f.write(f"{'='*60}\n")
    for i, row in feature_imp.iterrows():
        f.write(f"{row['feature']:20s} : {row['importance']:.4f}\n")

print("✓ Feature info saved to 'models/copd_features.txt'")

# Create a sample input template
sample_input = pd.DataFrame([X.iloc[0]], columns=feature_columns)
sample_input.to_csv('models/sample_input.csv', index=False)
print("✓ Sample input template saved to 'models/sample_input.csv'")

print(f"\n{'='*60}")
print("TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"✓ Model Accuracy: {accuracy*100:.2f}%")
if roc_auc:
    print(f"✓ ROC-AUC Score: {roc_auc:.4f}")
print(f"✓ All files saved in 'models/' directory")
print(f"✓ Model is ready for deployment!")
print(f"\nTo use the model for prediction:")
print(f"  1. Load the model: pickle.load(open('models/copd_model.pkl', 'rb'))")
print(f"  2. Load the scaler: pickle.load(open('models/copd_scaler.pkl', 'rb'))")
print(f"  3. Scale input: scaler.transform(input_data)")
print(f"  4. Predict: model.predict(scaled_data)")