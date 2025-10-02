import pandas as pd
import joblib
import os

# Load the CSV dataset
df = pd.read_csv("alzheimers_prediction_dataset.csv")

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the dataframe as a .sav file using joblib
joblib.dump(df, "models/alzheimers_prediction_dataset.sav")

print(f"Dataset saved successfully!")
print(f"Shape: {df.shape}")
print(f"File location: models/alzheimers_prediction_dataset.sav")

# Optional: Verify by loading it back
loaded_df = joblib.load("models/alzheimers_prediction_dataset.sav")
print(f"\nVerification - Loaded shape: {loaded_df.shape}")
print("First few rows:")
print(loaded_df.head())