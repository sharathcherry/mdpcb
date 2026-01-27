"""
Chronic Kidney Disease Prediction Model
Using Gradient Boosting Classifier - 92.77% CV Accuracy
Predicts CKD diagnosis based on comprehensive health markers
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print('='*70)
print('CHRONIC KIDNEY DISEASE PREDICTION - GRADIENT BOOSTING')
print('='*70)

# Load data
df = pd.read_csv('train/chronic_kidney_disease.csv')
print(f'Original shape: {df.shape}')

# Target
print('Target distribution:')
print(df['Diagnosis'].value_counts())
print()

# Drop non-useful columns
drop_cols = ['PatientID', 'DoctorInCharge']
X = df.drop(['Diagnosis'] + drop_cols, axis=1)
y = df['Diagnosis']

# Save feature columns
feature_columns = list(X.columns)
numerical_columns = list(X.select_dtypes(include=[np.number]).columns)
categorical_columns = list(X.select_dtypes(include=['object']).columns)

print(f'Total features: {len(feature_columns)}')
print(f'Numerical features: {len(numerical_columns)}')
print(f'Categorical features: {len(categorical_columns)}')
print()

# Fill NaN with median
X = X.fillna(X.median())

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f'Train size: {len(X_train)}, Test size: {len(X_test)}')

# Train Gradient Boosting model
print('\nTraining Gradient Boosting Classifier...')
model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
print()
print('='*70)
print('RESULTS')
print('='*70)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {test_accuracy*100:.2f}%')
print()
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=['No CKD', 'CKD Positive']))
print()
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# CV scores
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
print()
print(f'CV Scores: {[round(s, 4) for s in cv_scores]}')
print(f'CV Mean Accuracy: {cv_scores.mean()*100:.2f}%')

# Feature importance
print()
print('Top 15 Features:')
importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
feature_importance = importances.to_dict()
for feat, imp in importances.head(15).items():
    print(f'  {feat}: {imp:.4f}')

# Retrain on full data
print('\n' + '='*70)
print('Retraining on full dataset...')
model_final = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model_final.fit(X_scaled, y)

# Save packaged model
model_package = {
    'model': model_final,
    'scaler': scaler,
    'encoders': {},
    'feature_columns': feature_columns,
    'numerical_columns': numerical_columns,
    'categorical_columns': categorical_columns,
    'accuracy': test_accuracy,
    'cv_accuracy': cv_scores.mean(),
    'model_type': 'Gradient Boosting',
    'feature_importance': feature_importance,
    'classes': ['No CKD', 'CKD Positive'],
    'description': 'Chronic Kidney Disease Prediction - Comprehensive health marker analysis'
}

# Save model
os.makedirs('new/models', exist_ok=True)
model_path = 'new/models/chronic_model.sav'
with open(model_path, 'wb') as f:
    pickle.dump(model_package, f)

print(f'\nModel saved to: {model_path}')
print(f'Final Test Accuracy: {test_accuracy*100:.2f}%')
print(f'Final CV Accuracy: {cv_scores.mean()*100:.2f}%')

# Verify
print('\n' + '='*70)
print('VERIFICATION')
print('='*70)
loaded = pickle.load(open(model_path, 'rb'))
print(f"Model type: {loaded['model_type']}")
print(f"Accuracy: {loaded['accuracy']*100:.2f}%")
print(f"CV Accuracy: {loaded['cv_accuracy']*100:.2f}%")
print(f"Features: {len(loaded['feature_columns'])}")
print(f"Classes: {loaded['classes']}")
