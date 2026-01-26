import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_save_and_evaluate(filename, model_filename):
    try:
        df = pd.read_csv(filename)
        df = df.dropna()
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category').cat.codes
        if y.dtype == 'object':
            y = y.astype('category').cat.codes
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)
        
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        joblib.dump(clf, model_filename)

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained, saved to {model_filename}. Accuracy: {accuracy:.4f}")
        return accuracy
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

if __name__ == "__main__":
    datasets = {
        "heart.csv": "heart_model.sav",
        "migraine_data.csv": "migraine_model.sav",
        "lung_cancer.csv": "lung_cancer_model.sav",
        "pneumonia_dataset.csv": "pneumonia_model.sav",
        "alzheimers_disease_data.csv": "alzheimers_model.sav"
    }
    
    accuracies = {}
    for csv_file, model_file in datasets.items():
        accuracy = train_save_and_evaluate(csv_file, model_file)
        accuracies[csv_file] = accuracy
    
    print("\nFinal accuracies:")
    for file, acc in accuracies.items():
        if acc is not None:
            print(f"{file}: {acc:.4f}")
        else:
            print(f"{file}: Error during training")
