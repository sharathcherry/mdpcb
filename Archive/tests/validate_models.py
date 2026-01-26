import pickle
import os
import sys

# Model mapping from app.py
MODEL_FILES = {
    "diabetes_model": "models/diabetes_model.sav",
    "heart_model": "models/heart_disease_model.sav",
    "parkinson_model": "models/parkinsons_model.sav",
    "lung_cancer_model": "models/lung_cancer_model.sav",
    "breast_cancer_model": "models/breast_cancer.sav",
    "kidney": "models/kidney_disease.sav",
    "hepatitis_model": "models/hepititisc_model.sav",
    "liver_model": "models/liver_model.sav",
    "alzheimers_model": "models/alzheimers_model.sav",
    "epilepsy_model": "models/epilepsy_model.sav",
    "migraine_model": "models/migraine_model.sav",
    "tb_model": "models/tuberculosis_model.sav",
    "hiv_model": "models/hiv_model.sav",
    "malaria_model": "models/malaria_model.sav",
    "colorectal_model": "models/colorectal_model.sav",
    "prostate_model": "models/prostate_model.sav",
    "cervical_model": "models/cervical_model.sav",
    "asthma_model": "models/asthma_model.sav",
    "copd_model": "models/copd_model.sav",
    "pneumonia_model": "models/pneumonia_model.sav",
    "obesity_model": "models/obesity_model.sav",
}

def validate_models():
    print(f"{'Model Name':<25} | {'Status':<15} | {'Features':<10} | {'Type'}")
    print("-" * 75)
    
    results = []
    
    for name, path in MODEL_FILES.items():
        full_path = os.path.join(os.getcwd(), path)
        status = "OK"
        features = "N/A"
        model_type = "Unknown"
        
        if not os.path.exists(full_path):
            status = "MISSING"
        else:
            try:
                with open(full_path, "rb") as f:
                    model = pickle.load(f)
                
                model_type = type(model).__name__
                
                if hasattr(model, "predict"):
                    # Check for sklearn feature count
                    if hasattr(model, "n_features_in_"):
                        features = str(model.n_features_in_)
                    elif hasattr(model, "coef_"):
                        features = str(model.coef_.shape[1])
                    else:
                        features = "Found"
                else:
                    status = "NO PREDICT"
            except Exception as e:
                status = f"ERROR: {str(e)[:20]}"
        
        print(f"{name:<25} | {status:<15} | {features:<10} | {model_type}")
        results.append((name, status))

    print("-" * 75)
    total = len(results)
    passed = sum(1 for _, s in results if s == "OK")
    print(f"Summary: {passed}/{total} models loaded successfully.")

if __name__ == "__main__":
    validate_models()
