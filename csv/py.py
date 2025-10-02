import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
df = pd.read_csv("alzheimers_prediction_dataset.csv")

# Separate features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Impute missing data for categorical columns
cat_cols = X.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])

# Impute missing data for numeric columns
num_cols = X.select_dtypes(include=['number']).columns
if len(num_cols) > 0:
    imputer_num = SimpleImputer(strategy='mean')
    X[num_cols] = imputer_num.fit_transform(X[num_cols])

# Store label encoders
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*50}")
print(f"Model Accuracy: {accuracy:.4f}")
print(f"{'='*50}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# Save model and preprocessors
if accuracy >= 0.7:
    os.makedirs("models", exist_ok=True)
    
    # Save using .sav extension
    joblib.dump(model, "models/alzheimers_model.sav")
    joblib.dump(scaler, "models/scaler.sav")
    joblib.dump(label_encoders, "models/label_encoders.sav")
    joblib.dump(target_encoder, "models/target_encoder.sav")
    joblib.dump(list(X.columns), "models/feature_names.sav")
    
    print("\n✓ Model and preprocessors saved successfully!")
    print("Files created:")
    print("  - models/alzheimers_model.sav")
    print("  - models/scaler.sav")
    print("  - models/label_encoders.sav")
    print("  - models/target_encoder.sav")
    print("  - models/feature_names.sav")
else:
    print("\n✗ Model accuracy too low, not saving")