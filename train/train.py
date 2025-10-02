# Example for diabetes model training (create a separate script)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # or whatever model you want
import joblib

# Load your dataset
data = pd.read_csv('diabetes.csv')

# Separate features and target
X = data.drop('Outcome', axis=1)  # adjust column name
y = data['Outcome']

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the TRAINED MODEL (not the data)
joblib.dump(model, 'models/diabetes_model.sav')