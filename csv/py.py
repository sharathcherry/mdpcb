import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
df = pd.read_csv("alzheimers_prediction_dataset.csv")

# Separate features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Impute missing data for categorical columns if any
cat_cols = X.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])

# Impute missing data for numeric columns if any
num_cols = X.select_dtypes(include=['number']).columns
if len(num_cols) > 0:
    imputer_num = SimpleImputer(strategy='mean')
    X[num_cols] = imputer_num.fit_transform(X[num_cols])

# Encode categorical features after imputation
for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Encode target if categorical
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

if(accuracy >=0.7):# Save trained model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/alzheimers_disease_data.sav")
    print("done")
else:
    print("not done")