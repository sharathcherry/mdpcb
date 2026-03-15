# 🏥 Multiple Disease Prediction System (MDP)

> An AI-powered, full-stack health intelligence platform that predicts **23 diseases** using trained machine learning models and delivers personalized care plans powered by **NVIDIA's Llama 3.3 70B** LLM — with a RAG knowledge base, user authentication, and downloadable health reports.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Disease Coverage](#-disease-coverage)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Getting Started](#-getting-started)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [RAG Knowledge Base](#-rag-knowledge-base)
- [Training Your Own Models](#-training-your-own-models)
- [Contributing](#-contributing)
- [Disclaimer](#-disclaimer)

---

## 🔍 Overview

The **Multiple Disease Prediction System** is a research-grade web application that combines traditional machine learning with modern LLM capabilities. A user enters their clinical parameters (lab values, symptoms, demographics), the system runs a scikit-learn model to predict disease risk, and then an LLM generates a fully structured health management plan covering diet, medications, doctor referral urgency, precautions, and exercise — all within seconds.

The platform is split into two concurrently running services:

| Service | Technology | Port | Role |
|---------|-----------|------|------|
| **Frontend** | Streamlit | 8501 | Interactive UI, model calls, auth |
| **Backend** | Flask REST API | 5000 | Model inference, auth, SQLite persistence |

---

## ✨ Features

### 🤖 AI & Machine Learning
- **23 disease prediction models** — all trained with scikit-learn, serialized in `.sav` packages that include the model, scaler, and encoders
- **Symptom-based general prediction** — enter any combination of symptoms to identify the most likely disease from 100+ categories
- **Confidence scores & probabilities** — every prediction returns class probabilities for transparency
- **RAG-enhanced LLM recommendations** — Retrieval-Augmented Generation grounds advice in uploaded medical PDFs before hitting the LLM

### 💊 Personalized Health Plans (LLM-powered)
- **Dietary plan** — calorie targets, macros, meal schedule, vitamins & minerals, portion guides
- **Medication details** — prescription vs OTC options, dosage, frequency, duration, cost, generic alternatives
- **Doctor referral** — urgency level (immediate → routine), specialist type, recommended tests, follow-up schedule
- **Precautions** — lifestyle changes, activities to avoid, warning signs, emergency symptoms
- **Exercise recommendations** — type, duration, frequency, intensity
- **Downloadable PDF/Text health report**

### 👤 User Management
- JWT-based authentication (register, login, session management)
- User profile (age, gender, height, weight, blood group, medical history)
- Persistent prediction history per user
- Auto session-expiry with graceful logout

### 📊 Analytics & Transparency
- **Model Metrics dashboard** — accuracy, precision, recall, F1-score, ROC-AUC for every loaded model
- Feature importance tables (top features per model)
- Configuration details (algorithm type, estimators, feature count, classes)

---

## 🩺 Disease Coverage

| Category | Diseases |
|----------|---------|
| **Metabolic** | Diabetes (Type 2), Obesity |
| **Cardiovascular** | Heart Disease |
| **Neurological** | Parkinson's, Alzheimer's, Epilepsy, Migraine |
| **Organ** | Liver Disease, Liver Cancer, Chronic Kidney Disease |
| **Infectious** | Hepatitis C, Tuberculosis, Malaria |
| **Cancer** | Lung Cancer, Breast Cancer, Prostate Cancer, Cervical Cancer, Cancer Risk Assessment |
| **Respiratory** | Asthma, COPD, Pneumonia |
| **General** | General Disease Prediction (symptom-based, 100+ diseases) |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND (port 8501)               │
│  new/app.py — 23 disease prediction UIs + auth + history       │
│  ┌──────────┐  ┌─────────────┐  ┌──────────────────────────┐  │
│  │  Disease  │  │  RAG Engine │  │  UI Reports (ui/reports) │  │
│  │   Forms   │  │ (rag_engine)│  │  Tabs: Diet/Meds/Doctor  │  │
│  └─────┬────┘  └──────┬──────┘  └──────────── ─────────────┘  │
│        │              │                                          │
│        └──────────────┼──────────────────────────────────────  │
│                       ▼                                          │
│              HTTP Requests (requests lib)                        │
└───────────────────────────────────────────────────────────────  ┘
                        │
                        ▼  REST API  (port 5000)
┌─────────────────────────────────────────────────────────────────┐
│                    FLASK BACKEND (new/server.py)                 │
│  ┌──────────────┐  ┌────────────────┐  ┌───────────────────┐  │
│  │  ML Models   │  │  Auth + JWT    │  │  LLM Proxy        │  │
│  │  (/predict/*)│  │  (/api/auth/*) │  │  (/api/llm/*)     │  │
│  └──────┬───────┘  └────────┬───────┘  └────────┬──────────┘  │
│         │                   │                    │              │
│  ┌──────▼───────┐   ┌───────▼─────────┐  ┌──────▼──────────┐ │
│  │  23 × .sav   │   │  SQLite DB      │  │  NVIDIA API     │ │
│  │  pickled     │   │  Users/Profiles │  │  Llama 3.3 70B  │ │
│  │  model pkgs  │   │  Predictions    │  │  (streaming)    │ │
│  └──────────────┘   └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### RAG Pipeline (optional, enhances LLM results)

```
User Input → Disease Query
      │
      ▼
rag_engine.retrieve_context()
      │
      ├── Vector search in rag_documents/ (PDFs)
      │
      ▼
retrieved medical passages (chunks)
      │
      ▼
build_grounded_prompt() → embed into LLM prompt
      │
      ▼
NVIDIA Llama 3.3 70B → structured JSON response
      │
      ▼
ui/reports.py → tabbed Streamlit display
```

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | [Streamlit](https://streamlit.io) |
| **Backend API** | [Flask](https://flask.palletsprojects.com) + Flask-JWT-Extended + Flask-Bcrypt |
| **Database** | SQLite (via Flask-SQLAlchemy) |
| **ML Models** | scikit-learn (Random Forest, SVM, etc.) |
| **LLM** | NVIDIA NIM — `meta/llama-3.3-70b-instruct` (OpenAI-compatible API) |
| **RAG** | Custom `rag_engine.py` module |
| **Data** | NumPy, Pandas, Pillow |
| **Auth** | JWT (Bearer tokens, 24-hour expiry) |
| **Styling** | Custom CSS (`ui/style.css`) |

---

## 📁 Project Structure

```
mdpcblatest/
│
├── new/                          # Primary application (active)
│   ├── app.py                    # Main Streamlit frontend (6 500+ lines)
│   ├── server.py                 # Flask REST API backend (769 lines)
│   ├── models.py                 # SQLAlchemy ORM models (User, Profile, Prediction)
│   ├── rag_engine.py             # RAG pipeline (retrieve, classify, ground prompts)
│   ├── models/                   # Trained ML model files (.sav)
│   │   ├── diabetes_model.sav
│   │   ├── heart_disease_model.sav
│   │   ├── breast_cancer.sav
│   │   ├── kidney_disease_model.sav
│   │   ├── lung_cancer_model.sav
│   │   ├── parkinsons_model.sav
│   │   ├── liver_cancer_model.sav
│   │   ├── hepatitis_c_model.sav
│   │   ├── asthma_model.sav
│   │   ├── malaria_model.sav
│   │   ├── alzheimers_model.sav
│   │   ├── epilepsy_model.sav
│   │   ├── obesity_model.sav
│   │   ├── prostate_model.sav
│   │   ├── cancer_risk_model.sav
│   │   ├── migraine_model.sav
│   │   ├── tuberculosis_model.sav
│   │   ├── copd_model.sav
│   │   ├── cervical_model.sav
│   │   ├── chronic_model.sav
│   │   ├── liver_disease_model.sav
│   │   ├── pneumonia_model.sav
│   │   └── general_disease_model.sav
│   ├── services/                 # Service layer (empty — LLM logic lives in services/llm.py)
│   └── train_liver.py            # Liver model training script
│
├── ui/
│   ├── reports.py                # Streamlit report rendering (tabs, metrics, download)
│   └── style.css                 # Custom CSS theme
│
├── services/
│   ├── llm.py                    # LLM helper (NVIDIA API calls, JSON cleaning)
│   └── model_loader.py           # Utility to load .sav packages
│
├── train/                        # Dataset CSVs and training scripts
│   ├── cancer_patient_data.csv
│   ├── chronic_kidney_disease.csv
│   ├── hepatitis_c.csv
│   ├── liver_cancer.csv
│   ├── lung_cancer.csv
│   ├── parkinsons_disease_data.csv
│   ├── train_cancer_risk.py
│   ├── train_epilepsy.py
│   ├── train_liver_cancer.py
│   ├── train_prostate.py
│   └── train_tuberculosis.py
│
├── app.py                        # Legacy Streamlit entry (root-level, references new/)
├── constants.py                  # Shared constants (icons, model paths, health tips)
├── requirements.txt              # Python dependencies
├── build_notebook.py             # Jupyter notebook builder
├── update_all_model_metrics.py   # Bulk model metrics updater
├── verify_models.py              # Model integrity checker
├── model_validation_report.txt   # Accuracy report for all models
├── research_graphs.ipynb         # Research/EDA notebook
├── architecture.html             # Visual architecture diagram
├── RAG_PDF_COLLECTION_GUIDE.md   # Guide for building RAG knowledge base
└── .streamlit/                   # Streamlit config & secrets
```

---

## 📈 Model Performance

All models are packaged as dictionaries containing `{model, scaler, encoders, metrics}`.

| Disease | Algorithm | Accuracy | Precision | Recall | F1-Score |
|---------|-----------|----------|-----------|--------|----------|
| 🎗️ Breast Cancer | Random Forest | **97.37%** | 95.56% | 97.73% | 96.63% |
| 🫁 Pneumonia | Random Forest | **98.59%** | — | — | — |
| 🧪 Hepatitis C | Random Forest | 96.75% | 96.30% | 97.36% | 96.82% |
| 🔬 Liver Cancer | Random Forest | 94.30% | 95.00% | 93.42% | 94.20% |
| 🌫️ Asthma | — | 94.78% | 40.00% | 4.00% | 7.27%* |
| 🦟 Malaria | — | 94.77% | 96.00% | 97.00% | 96.00% |
| 🧩 Alzheimer's | Random Forest (500 trees) | 94.42% | 94.00% | 89.00% | 92.00% |
| 🧠 Parkinson's | — | 92.64% | 92.31% | 95.16% | 93.71% |
| 🫘 Kidney Disease | — | 92.17% | 92.50% | 80.43% | 86.05% |
| 🩸 Diabetes | — | 90.48% | 89.47% | 91.43% | 90.44% |
| ❤️ Heart Disease | — | 89.13% | 88.89% | 84.21% | 86.49% |
| 🌬️ Lung Cancer | — | **100.00%** | 100.00% | 100.00% | 100.00% |

> ⚠️ *Asthma model has high overall accuracy but poor positive-class detection due to severe class imbalance (only 5.18% positive cases in training data).

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip
- (Optional) NVIDIA API key for LLM features — free at [build.nvidia.com](https://build.nvidia.com)

### 1. Clone the repository

```bash
git clone https://github.com/sharathcherry/mdpcb.git
cd mdpcb
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up secrets (optional — for LLM features)

Create `.streamlit/secrets.toml`:

```toml
NVIDIA_API_KEY = "nvapi-xxxxxxxxxxxxxxxxxxxx"
```

Or set it as an environment variable:

```bash
export NVIDIA_API_KEY="nvapi-xxxxxxxxxxxxxxxxxxxx"
```

### 5. Start the Flask backend

```bash
cd new
python server.py
# Server starts on http://localhost:5000
```

### 6. Start the Streamlit frontend (new terminal)

```bash
cd new
streamlit run app.py
# App opens on http://localhost:8501
```

---

## ⚙️ Configuration

| Variable | Where | Description |
|----------|-------|-------------|
| `NVIDIA_API_KEY` | `.streamlit/secrets.toml` or env | NVIDIA NIM API key for LLM recommendations |
| `API_BASE_URL` | `new/app.py` line 13 | Flask API URL (default: `http://localhost:5000/api`) |
| `JWT_SECRET_KEY` | `new/server.py` line 23 | Change in production! |
| `SQLALCHEMY_DATABASE_URI` | `new/server.py` line 21 | SQLite path (default: `new/instance/health_system.db`) |
| `JWT_ACCESS_TOKEN_EXPIRES` | `new/server.py` | Token TTL (default: 24 hours) |

---

## 🌐 API Reference

The Flask backend exposes a REST API on port **5000**.

### Health Check

```
GET /api/health
Response: { "status": "ok", "models_loaded": 23, "models_failed": 0 }
```

### Authentication

| Method | Endpoint | Payload |
|--------|----------|---------|
| `POST` | `/api/auth/register` | `{ username, email, password }` |
| `POST` | `/api/auth/login` | `{ username, password }` → returns `access_token` |

### Disease Prediction Endpoints

| Endpoint | Disease | Key Inputs |
|----------|---------|------------|
| `POST /api/predict/diabetes` | Diabetes | gender, age, bmi, hba1c, blood_glucose, hypertension... |
| `POST /api/predict/heart` | Heart Disease | age, sex, chest_pain_type, resting_bp, cholesterol... |
| `POST /api/predict/parkinsons` | Parkinson's | `{ features: [...22 voice measurements] }` |
| `POST /api/predict/lung-cancer` | Lung Cancer | age, gender, air_pollution, smoking, genetic_risk... |
| `POST /api/predict/breast-cancer` | Breast Cancer | `{ features: [...30 cell nucleus measurements] }` |
| `POST /api/predict/liver-cancer` | Liver Cancer | age, gender, bmi, alcohol_consumption, smoking_status... |
| `POST /api/predict/kidney` | Kidney Disease | `{ features: [...] }` |
| `POST /api/predict/liver-disease` | Liver Disease | age, gender, bilirubin, aminotransferases, albumin... |
| `POST /api/predict/hepatitis` | Hepatitis C | age, sex, ALB, ALP, ALT, AST, BIL, CHE... |
| `POST /api/predict/general` | General (symptom) | `{ symptoms: ["fever", "cough", ...] }` |
| `POST /api/predict/<model_key>` | Any model | `{ features: [...] }` generic fallback |

### LLM Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/llm/recommendations` | Get full care plan (diet, meds, doctor, exercise) |
| `POST` | `/api/llm/health-tips` | Get disease-specific health tips |
| `GET` | `/api/llm/general-tips` | Get general wellness tips |

### User Data

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET/PUT` | `/api/user/profile` | View or update user profile (JWT required) |
| `GET` | `/api/user/history` | List all past predictions (JWT required) |
| `POST` | `/api/user/history/save` | Save a new prediction result (JWT required) |

### Model Metrics

```
GET /api/models/metrics
Response: { "models": { "diabetes_model": { accuracy, precision, recall, f1, ... } } }
```

---

## 📚 RAG Knowledge Base

The system supports Retrieval-Augmented Generation to ground LLM advice in real medical literature.

### Setup

1. Create the `rag_documents/` folder in the project root
2. Add disease-specific medical PDFs in subfolders (see `RAG_PDF_COLLECTION_GUIDE.md`)
3. Run the indexing script (or the app will index on first launch)
4. The RAG status indicator in the sidebar shows 🟢 when documents are loaded

### Recommended PDF Sources

| Source | URL |
|--------|-----|
| WHO Publications | https://www.who.int/publications |
| American Diabetes Association | https://diabetesjournals.org/care |
| NCCN Patient Guidelines | https://www.nccn.org/patients |
| GINA (Asthma) | https://ginasthma.org |
| GOLD (COPD) | https://goldcopd.org |
| KDIGO (Kidney) | https://kdigo.org/guidelines |
| PubMed Central | https://www.ncbi.nlm.nih.gov/pmc |

> See `RAG_PDF_COLLECTION_GUIDE.md` for the full list of 88 recommended PDFs across all 22 disease categories.

---

## 🔬 Training Your Own Models

Training scripts are in the `train/` directory. Each script follows the same pattern:

```bash
python train/train_liver_cancer.py
# Outputs: new/models/liver_cancer_model.sav
```

**Model Package Format** — every `.sav` file is pickled as:

```python
{
    "model": <trained sklearn estimator>,
    "scaler": <fitted StandardScaler or None>,
    "label_encoder": <LabelEncoder or None>,
    "accuracy": 0.9437,
    "precision_weighted": ...,
    "recall_weighted": ...,
    "f1_weighted": ...,
    "roc_auc_weighted": ...,
    "feature_importance": [...],
    "feature_columns": [...],
    "n_features": 23,
    "model_type": "RandomForestClassifier"
}
```

To retrain all models and update their metrics:

```bash
python update_all_model_metrics.py
```

To verify model integrity after training:

```bash
python verify_models.py
```

---

## 🗄️ Database Schema

SQLite database at `new/instance/health_system.db`

| Table | Columns |
|-------|---------|
| `users` | id, username, email, password\_hash, created\_at |
| `profiles` | id, user\_id (FK), full\_name, age, gender, height, weight, blood\_group, medical\_history (JSON), updated\_at |
| `predictions` | id, user\_id (FK, nullable), disease, input\_data (JSON), result (JSON), created\_at |

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add new disease model"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

### Ideas for Contribution

- [ ] Add more disease models (HIV/AIDS, Colorectal Cancer)
- [ ] Build and populate the RAG knowledge base (PDFs)
- [ ] Add multi-language support for health tips
- [ ] Dockerize the full stack
- [ ] Add real-time vitals monitoring integration
- [ ] Implement model explainability (SHAP values visualization)

---

## ⚠️ Disclaimer

> **This application is for educational and research purposes only.**
> 
> The disease predictions and health recommendations generated by this system are **not** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
> 
> - Never disregard professional medical advice because of something you have read or seen in this application.
> - If you think you may have a medical emergency, call your doctor or emergency services immediately.
> - The AI-generated care plans are informational only and must be reviewed by a qualified healthcare professional before acting on them.

---

## 📄 License

This project is for academic use. See the repository owner for licensing details.

---

<div align="center">
  <strong>Built with ❤️ using Streamlit, Flask, scikit-learn, and NVIDIA NIM</strong><br/>
  <em>Average model accuracy: 94%+ across 23 diseases</em>
</div>
