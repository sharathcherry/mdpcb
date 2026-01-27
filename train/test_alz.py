import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

print('ALZHEIMERS PREDICTION DATASET (74K)')

df = pd.read_csv('Archive/csv/alzheimers_prediction_dataset.csv')
print('Shape:', df.shape)

# Target column
target_col = "Alzheimer's Diagnosis"
y = (df[target_col] == 'Yes').astype(int)
X = df.drop(target_col, axis=1)

# Encode categorical
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sample for speed (use 10K)
from sklearn.utils import resample
idx = resample(range(len(X_scaled)), n_samples=10000, random_state=42)
X_sample = X_scaled[idx]
y_sample = y.iloc[idx]

print()
models = [
    ('Random Forest', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('Extra Trees', ExtraTreesClassifier(n_estimators=200, random_state=42)),
]

for name, model in models:
    scores = cross_val_score(model, X_sample, y_sample, cv=5, scoring='accuracy')
    print(f'{name}: {scores.mean()*100:.2f}%')
