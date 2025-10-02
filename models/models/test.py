import joblib
import numpy as np
import os

# --- Configuration ---
MODEL_FILENAME = '"C:\\Users\\katuk\\OneDrive\\Desktop\\MDP\\models\\alzheimers_model.sav"'

def test_sav_file_functionality():
    """
    Tests a .sav model file by loading it and using it for a single prediction.
    This does NOT measure accuracy but confirms the file is valid and functional.
    """
    print(f"--- Starting Test for '{MODEL_FILENAME}' ---")

    # --- Test 1: Check if the file exists and can be loaded ---
    print("\n[Test 1: Loading the model file]")
    if not os.path.exists(MODEL_FILENAME):
        print(f"--- ERROR: Model file not found! ---")
        print(f"Please make sure '{MODEL_FILENAME}' is in the same folder as this script.")
        return

    try:
        # joblib.load is the standard way to load models saved with scikit-learn
        model = joblib.load(MODEL_FILENAME)
        print("✅ SUCCESS: Model file loaded without errors.")
    except Exception as e:
        print(f"--- ❌ FAILURE: Could not load the model file. ---")
        print(f"The file might be corrupted or was not saved correctly.")
        print(f"Error details: {e}")
        return

    # --- Test 2: Inspect the loaded model ---
    print("\n[Test 2: Inspecting the model's properties]")
    print(f"The loaded object is a: {type(model)}")
    print("Model Parameters:")
    print(model)

    # --- Test 3: Perform a single prediction (Functional Test) ---
    print("\n[Test 3: Making a prediction with sample data]")
    
    # Create a sample data point. This must have the same number of features
    # that the model was trained on, in the same order.
    # The features for the Pima Diabetes dataset are typically:
    # [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    
    # Example data for a hypothetical patient
    sample_patient_data = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]
    
    print(f"Input data for prediction: {sample_patient_data}")

    try:
        # Use the loaded model to predict the outcome for our sample data
        # The .predict() method expects a 2D array-like input, hence the double brackets [[...]]
        prediction = model.predict(sample_patient_data)
        
        # The .predict_proba() method can show the confidence of the prediction (if the model supports it)
        try:
            prediction_probability = model.predict_proba(sample_patient_data)
            confidence = prediction_probability[0][prediction[0]] * 100
            print(f"✅ SUCCESS: Model made a prediction.")
            print(f"-> Prediction Result: {prediction[0]} (0 = No Diabetes, 1 = Has Diabetes)")
            print(f"-> Prediction Confidence: {confidence:.2f}%")
        except AttributeError:
            # Some models (like basic Linear Regression) don't have predict_proba
            print(f"✅ SUCCESS: Model made a prediction.")
            print(f"-> Prediction Result: {prediction[0]} (0 = No Diabetes, 1 = Has Diabetes)")
            print("(This model does not provide prediction probabilities.)")

    except Exception as e:
        print(f"--- ❌ FAILURE: The model failed to make a prediction. ---")
        print("This often happens if the input data does not have the correct number of features.")
        print(f"Error details: {e}")
        return
        
    print("\n--- All Tests Complete ---")
    print("Conclusion: The .sav file is a valid and functional model.")


if __name__ == '__main__':
    test_sav_file_functionality()