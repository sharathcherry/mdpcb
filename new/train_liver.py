"""
Liver Disease Prediction Model Training Script
Dataset: Indian Liver Patient Dataset (ILPD)
Source: indian_liver_patient.csv from Downloads folder
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(filepath):
    """Load and preprocess the liver disease dataset"""
    print("=" * 60)
    print("LIVER DISEASE MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"\n📊 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Display columns
    print(f"\n📋 Features: {list(df.columns)}")
    
    # Check target distribution (Dataset column: 1 = Liver Patient, 2 = Non-Liver Patient)
    print(f"\n🎯 Target Distribution (Original):")
    print(df['Dataset'].value_counts())
    
    # Convert target: 1 (Liver) -> 1, 2 (Non-Liver) -> 0
    df['Target'] = df['Dataset'].apply(lambda x: 1 if x == 1 else 0)
    print(f"\n🎯 Target Distribution (Converted - 1=Liver Patient, 0=Healthy):")
    print(df['Target'].value_counts())
    print(f"Balance: {df['Target'].value_counts(normalize=True).to_dict()}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️ Missing values found:")
        print(missing[missing > 0])
        # Fill missing values with median
        df['Albumin_and_Globulin_Ratio'].fillna(
            df['Albumin_and_Globulin_Ratio'].median(), inplace=True
        )
        print("✅ Missing values filled with median")
    else:
        print(f"\n✅ No missing values!")
    
    # Encode Gender: Male = 1, Female = 0
    gender_mapping = {'Male': 1, 'Female': 0}
    df['Gender_encoded'] = df['Gender'].map(gender_mapping)
    print(f"\n🔄 Gender encoding: {gender_mapping}")
    
    # Define feature columns
    feature_columns = [
        'Age',
        'Gender_encoded',
        'Total_Bilirubin',
        'Direct_Bilirubin',
        'Alkaline_Phosphotase',
        'Alamine_Aminotransferase',
        'Aspartate_Aminotransferase',
        'Total_Protiens',
        'Albumin',
        'Albumin_and_Globulin_Ratio'
    ]
    
    X = df[feature_columns]
    y = df['Target']
    
    print(f"\n📐 Feature matrix shape: {X.shape}")
    
    return X, y, gender_mapping, feature_columns


def train_model(X, y):
    """Train the liver disease prediction model"""
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📊 Train set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with optimized parameters
    print("\n🌲 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Cross-validation
    print("\n📈 Performing 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    print(f"   CV Scores: {cv_scores}")
    print(f"   CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%")
    print(f"✅ ROC-AUC Score: {roc_auc:.4f}")
    
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Liver Disease']))
    
    print(f"\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    print(f"\n🔍 Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in feature_importance.iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"   {row['feature']:30s}: {row['importance']:.4f} {bar}")
    
    return model, scaler, accuracy, roc_auc


def save_model(model, scaler, gender_mapping, feature_columns, filepath):
    """Save model and preprocessing components"""
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    model_package = {
        'model': model,
        'scaler': scaler,
        'gender_mapping': gender_mapping,
        'feature_columns': feature_columns
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"✅ Model saved to: {filepath}")
    
    # Print mappings for reference
    print(f"\n📋 Gender Mapping (for app.py):")
    print(f"   {gender_mapping}")


def test_prediction(model, scaler, gender_mapping):
    """Test prediction with sample inputs"""
    print("\n" + "=" * 60)
    print("TEST PREDICTIONS")
    print("=" * 60)
    
    # Sample test cases
    test_cases = [
        {
            'name': 'High Risk Patient (Elevated Bilirubin)',
            'Age': 45, 'Gender': 'Male',
            'Total_Bilirubin': 10.9, 'Direct_Bilirubin': 5.5,
            'Alkaline_Phosphotase': 699, 'Alamine_Aminotransferase': 64,
            'Aspartate_Aminotransferase': 100, 'Total_Protiens': 7.5,
            'Albumin': 3.2, 'Albumin_and_Globulin_Ratio': 0.74
        },
        {
            'name': 'Low Risk Patient (Normal Values)',
            'Age': 35, 'Gender': 'Female',
            'Total_Bilirubin': 0.7, 'Direct_Bilirubin': 0.2,
            'Alkaline_Phosphotase': 187, 'Alamine_Aminotransferase': 16,
            'Aspartate_Aminotransferase': 18, 'Total_Protiens': 6.8,
            'Albumin': 3.3, 'Albumin_and_Globulin_Ratio': 0.9
        },
        {
            'name': 'Moderate Risk Patient',
            'Age': 52, 'Gender': 'Male',
            'Total_Bilirubin': 2.6, 'Direct_Bilirubin': 1.2,
            'Alkaline_Phosphotase': 415, 'Alamine_Aminotransferase': 407,
            'Aspartate_Aminotransferase': 576, 'Total_Protiens': 6.4,
            'Albumin': 3.2, 'Albumin_and_Globulin_Ratio': 1.0
        }
    ]
    
    for case in test_cases:
        name = case.pop('name')
        
        # Encode gender
        gender_encoded = gender_mapping[case['Gender']]
        
        # Prepare input
        encoded_input = [
            case['Age'],
            gender_encoded,
            case['Total_Bilirubin'],
            case['Direct_Bilirubin'],
            case['Alkaline_Phosphotase'],
            case['Alamine_Aminotransferase'],
            case['Aspartate_Aminotransferase'],
            case['Total_Protiens'],
            case['Albumin'],
            case['Albumin_and_Globulin_Ratio']
        ]
        
        # Scale and predict
        input_scaled = scaler.transform([encoded_input])
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        result = "🔴 LIVER DISEASE RISK" if prediction == 1 else "🟢 HEALTHY"
        print(f"\n{name}:")
        print(f"   Prediction: {result}")
        print(f"   Risk Score: {probability*100:.1f}%")


if __name__ == "__main__":
    # Paths
    DATA_PATH = r"C:\Users\katuk\Downloads\indian_liver_patient.csv"
    MODEL_PATH = "models/liver_disease_model.sav"
    
    # Load and preprocess data
    X, y, gender_mapping, feature_columns = load_and_preprocess_data(DATA_PATH)
    
    # Train model
    model, scaler, accuracy, roc_auc = train_model(X, y)
    
    # Save model
    save_model(model, scaler, gender_mapping, feature_columns, MODEL_PATH)
    
    # Test predictions
    test_prediction(model, scaler, gender_mapping)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Final Model Performance:")
    print(f"   • Accuracy: {accuracy*100:.2f}%")
    print(f"   • ROC-AUC: {roc_auc:.4f}")
    print(f"\n✅ Model ready for integration into app.py")
