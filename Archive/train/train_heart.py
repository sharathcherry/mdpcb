"""
Heart Disease Prediction Model Training Script
Dataset: Heart Failure Prediction Dataset (fedesoriano - Kaggle)
Combined from 5 UCI datasets: Cleveland, Hungarian, Switzerland, Long Beach VA, Stalog
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """Load and preprocess the heart disease dataset"""
    print("="*60)
    print("HEART DISEASE MODEL TRAINING")
    print("="*60)
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"\n📊 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Display columns
    print(f"\n📋 Features: {list(df.columns)}")
    
    # Check target distribution
    print(f"\n🎯 Target Distribution:")
    print(df['HeartDisease'].value_counts())
    print(f"Balance: {df['HeartDisease'].value_counts(normalize=True).to_dict()}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️ Missing values found:")
        print(missing[missing > 0])
    else:
        print(f"\n✅ No missing values!")
    
    # Encode categorical variables
    encoders = {}
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    
    print(f"\n🔄 Encoding categorical features...")
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        encoders[col] = le
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"   {col}: {mapping}")
    
    # Define feature columns (using encoded versions)
    feature_columns = [
        'Age', 
        'Sex_encoded', 
        'ChestPainType_encoded', 
        'RestingBP', 
        'Cholesterol', 
        'FastingBS', 
        'RestingECG_encoded', 
        'MaxHR', 
        'ExerciseAngina_encoded', 
        'Oldpeak', 
        'ST_Slope_encoded'
    ]
    
    X = df[feature_columns]
    y = df['HeartDisease']
    
    print(f"\n📐 Feature matrix shape: {X.shape}")
    
    return X, y, encoders, feature_columns


def train_model(X, y):
    """Train the heart disease prediction model"""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
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
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%")
    print(f"✅ ROC-AUC Score: {roc_auc:.4f}")
    
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Heart Disease', 'Heart Disease']))
    
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
        print(f"   {row['feature']:25s}: {row['importance']:.4f} {bar}")
    
    return model, scaler, accuracy, roc_auc


def save_model(model, scaler, encoders, feature_columns, filepath):
    """Save model and preprocessing components"""
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    # Create mappings from encoders for easy lookup
    mappings = {}
    for col, encoder in encoders.items():
        mappings[col] = dict(zip(encoder.classes_.tolist(), 
                                  encoder.transform(encoder.classes_).tolist()))
    
    model_package = {
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'feature_columns': feature_columns,
        'mappings': mappings
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"✅ Model saved to: {filepath}")
    
    # Print mappings for reference
    print(f"\n📋 Encoding Mappings (for app.py):")
    for col, mapping in mappings.items():
        print(f"   {col}: {mapping}")


def test_prediction(model, scaler, encoders):
    """Test prediction with sample inputs"""
    print("\n" + "="*60)
    print("TEST PREDICTIONS")
    print("="*60)
    
    # Sample test cases
    test_cases = [
        {
            'name': 'High Risk Patient (Elderly, Asymptomatic chest pain)',
            'Age': 65, 'Sex': 'M', 'ChestPainType': 'ASY', 'RestingBP': 160,
            'Cholesterol': 286, 'FastingBS': 1, 'RestingECG': 'ST', 'MaxHR': 108,
            'ExerciseAngina': 'Y', 'Oldpeak': 2.5, 'ST_Slope': 'Flat'
        },
        {
            'name': 'Low Risk Patient (Young, Typical Angina)',
            'Age': 35, 'Sex': 'F', 'ChestPainType': 'TA', 'RestingBP': 120,
            'Cholesterol': 180, 'FastingBS': 0, 'RestingECG': 'Normal', 'MaxHR': 170,
            'ExerciseAngina': 'N', 'Oldpeak': 0.0, 'ST_Slope': 'Up'
        },
        {
            'name': 'Moderate Risk Patient',
            'Age': 52, 'Sex': 'M', 'ChestPainType': 'NAP', 'RestingBP': 140,
            'Cholesterol': 230, 'FastingBS': 0, 'RestingECG': 'Normal', 'MaxHR': 140,
            'ExerciseAngina': 'N', 'Oldpeak': 1.0, 'ST_Slope': 'Flat'
        }
    ]
    
    for case in test_cases:
        name = case.pop('name')
        
        # Encode categorical features
        encoded_input = [
            case['Age'],
            encoders['Sex'].transform([case['Sex']])[0],
            encoders['ChestPainType'].transform([case['ChestPainType']])[0],
            case['RestingBP'],
            case['Cholesterol'],
            case['FastingBS'],
            encoders['RestingECG'].transform([case['RestingECG']])[0],
            case['MaxHR'],
            encoders['ExerciseAngina'].transform([case['ExerciseAngina']])[0],
            case['Oldpeak'],
            encoders['ST_Slope'].transform([case['ST_Slope']])[0]
        ]
        
        # Scale and predict
        input_scaled = scaler.transform([encoded_input])
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        result = "🔴 HEART DISEASE RISK" if prediction == 1 else "🟢 LOW RISK"
        print(f"\n{name}:")
        print(f"   Prediction: {result}")
        print(f"   Risk Score: {probability*100:.1f}%")


if __name__ == "__main__":
    # Paths
    DATA_PATH = "csv/heart_failure_prediction.csv"
    MODEL_PATH = "models/heart_disease_model.sav"
    
    # Load and preprocess data
    X, y, encoders, feature_columns = load_and_preprocess_data(DATA_PATH)
    
    # Train model
    model, scaler, accuracy, roc_auc = train_model(X, y)
    
    # Save model
    save_model(model, scaler, encoders, feature_columns, MODEL_PATH)
    
    # Test predictions
    test_prediction(model, scaler, encoders)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📊 Final Model Performance:")
    print(f"   • Accuracy: {accuracy*100:.2f}%")
    print(f"   • ROC-AUC: {roc_auc:.4f}")
    print(f"\n✅ Model ready for integration into app.py")
