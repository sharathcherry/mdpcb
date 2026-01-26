import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
print("Loading dataset...")
df = pd.read_csv('alzheimers_prediction_dataset.csv')

# Find target column
target_col = None
for col in df.columns:
    if 'diagnosis' in col.lower():
        target_col = col
        break
if target_col is None:
    target_col = df.columns[-1]

# Load the saved encoders and model
print("Loading model and encoders...")
with open('models/alzheimers_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/alzheimers_label_encoders.pkl', 'rb') as f:
    encoder_data = pickle.load(f)
    categorical_encoders = encoder_data['categorical_encoders']
    target_encoder = encoder_data['target_encoder']
    feature_columns = encoder_data['feature_columns']

# Prepare data
df = df.dropna()
y = target_encoder.transform(df[target_col])
X = df[feature_columns].copy()

# Encode categorical features
for col in feature_columns:
    if col in categorical_encoders:
        X[col] = categorical_encoders[col].transform(X[col].astype(str))

# Split data (same random_state as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Evaluate on test set
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Additional metrics
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n{'='*50}")
print("SUMMARY METRICS")
print(f"{'='*50}")
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1-Score:  {f1*100:.2f}%")
print(f"{'='*50}\n")