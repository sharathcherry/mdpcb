# Multiple Disease Prediction Health Suite

This is a comprehensive health suite application built with Streamlit that leverages machine learning and a powerful AI to provide a wide range of health-related services. The application can predict the likelihood of 22 different diseases based on user-provided data and offers personalized health recommendations, diet plans, and lifestyle advice through an integrated AI Health Assistant.

## ✨ Features

*   **Multi-Disease Prediction**: Utilizes 22 different machine learning models to predict various diseases.
*   **AI Health Assistant**: An AI-powered assistant to provide personalized health recommendations, including diet plans, medication information, and exercise routines.
*   **Symptom-Based General Prediction**: Users can select their symptoms to get a general idea of potential underlying conditions.
*   **User-Friendly Interface**: A clean and intuitive interface built with Streamlit, organized by disease categories.
*   **Health Utilities**: Includes tools like a BMI calculator, appointment booking, and health reminders.
*   **Dynamic Health Tips**: Provides general and disease-specific health tips.

## 🩺 Diseases Covered

The application can predict the following diseases:

*   **Metabolic**:
    *   Diabetes
    *   Obesity
*   **Cardiovascular**:
    *   Heart Disease
*   **Neurological**:
    *   Parkinson's
    *   Alzheimer's
    *   Epilepsy
    *   Migraine
*   **Organ-Specific**:
    *   Liver Disease
    *   Kidney Disease
*   **Infectious**:
    *   Hepatitis C
    *   Tuberculosis (TB)
    *   HIV/AIDS
    *   Malaria
*   **Cancer**:
    *   Lung Cancer
    *   Breast Cancer
    *   Colorectal Cancer
    *   Prostate Cancer
    *   Cervical Cancer
*   **Respiratory**:
    *   Asthma
    *   COPD
    *   Pneumonia

## 🛠️ Technologies Used

*   **Backend**: Python
*   **Frontend**: Streamlit
*   **Machine Learning**: Scikit-learn, Joblib, Pandas, Numpy
*   **AI Integration**: NVIDIA NIMs
*   **Data Visualization**: Seaborn

## 🚀 Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Key:**
    Open `app.py` and replace the placeholder for `NVIDIA_API_KEY` with your actual NVIDIA API key.
    ```python
    NVIDIA_API_KEY = "YOUR_NVIDIA_API_KEY"
    ```

5.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 📖 Usage

1.  Launch the application using the command above.
2.  Use the sidebar to navigate to the desired section (e.g., "Home", "Diabetes Prediction", "AI Health Assistant").
3.  For disease prediction, input the required medical parameters in the provided fields and click the "Predict" button.
4.  For the AI Health Assistant, fill in the patient's details and the condition to receive personalized recommendations.
5.  Explore other features like "Health Tips", "Book Appointment", and "Set Reminder" through the sidebar.

## ⚠️ Disclaimer

This application is for informational and educational purposes only. The predictions and recommendations provided are not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read on this application.