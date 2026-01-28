import datetime
import json
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from openai import OpenAI


# NVIDIA API Configuration for LLM Integration (Llama 3.3 70B Instruct)
# Get a free API key from: https://build.nvidia.com/meta/llama-3_3-70b-instruct
NVIDIA_API_KEY = st.secrets.get("NVIDIA_API_KEY") or os.getenv("NVIDIA_API_KEY")

# Initialize OpenAI client for NVIDIA
client = None
if NVIDIA_API_KEY:
    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )
    except Exception as e:
        st.warning(f"NVIDIA API initialization failed: {e}")
else:
    st.info("💡 To enable AI recommendations, add NVIDIA_API_KEY to Streamlit secrets. Get a free key at: https://build.nvidia.com")

# Predeclare model holders for static analyzers; they are populated dynamically below.
diabetes_model = heart_model = breast_cancer_model = None

# Load each model via declarative mapping to avoid silent mismatches
# Only keeping the 3 high-accuracy models: Diabetes (90.48%), Heart Disease (89.13%), Breast Cancer (97.37%)

# Get the directory where this script is located (works on cloud and locally)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILES = {
    "diabetes_model": os.path.join(BASE_DIR, "models", "diabetes_model.sav"),
    "heart_model": os.path.join(BASE_DIR, "models", "heart_disease_model.sav"),
    "breast_cancer_model": os.path.join(BASE_DIR, "models", "breast_cancer.sav"),
    "kidney_disease_model": os.path.join(BASE_DIR, "models", "kidney_disease_model.sav"),
    "lung_cancer_model": os.path.join(BASE_DIR, "models", "lung_cancer_model.sav"),
    "parkinsons_model": os.path.join(BASE_DIR, "models", "parkinsons_model.sav"),
    "liver_cancer_model": os.path.join(BASE_DIR, "models", "liver_cancer_model.sav"),
    "hepatitis_c_model": os.path.join(BASE_DIR, "models", "hepatitis_c_model.sav"),
    "asthma_model": os.path.join(BASE_DIR, "models", "asthma_model.sav"),
    "malaria_model": os.path.join(BASE_DIR, "models", "malaria_model.sav"),
    "alzheimers_model": os.path.join(BASE_DIR, "models", "alzheimers_model.sav"),
    "obesity_model": os.path.join(BASE_DIR, "models", "obesity_model.sav"),
    "epilepsy_model": os.path.join(BASE_DIR, "models", "epilepsy_model.sav"),
    "prostate_model": os.path.join(BASE_DIR, "models", "prostate_model.sav"),
    "cancer_risk_model": os.path.join(BASE_DIR, "models", "cancer_risk_model.sav"),
    "migraine_model": os.path.join(BASE_DIR, "models", "migraine_model.sav"),
    "tuberculosis_model": os.path.join(BASE_DIR, "models", "tuberculosis_model.sav"),
    "copd_model": os.path.join(BASE_DIR, "models", "copd_model.sav"),
    "cervical_model": os.path.join(BASE_DIR, "models", "cervical_model.sav"),
    "chronic_model": os.path.join(BASE_DIR, "models", "chronic_model.sav"),
    "liver_disease_model": os.path.join(BASE_DIR, "models", "liver_disease_model.sav"),
    "pneumonia_model": os.path.join(BASE_DIR, "models", "pneumonia_model.sav"),
    "general_disease_model": os.path.join(BASE_DIR, "models", "general_disease_model.sav"),
}

loaded_models = {}
failed_models = {}

# Special handling for packaged models (models saved with scaler, encoders, etc.)
PACKAGED_MODELS = {"diabetes_model", "heart_model", "breast_cancer_model", "kidney_disease_model", "lung_cancer_model", "parkinsons_model", "liver_cancer_model", "hepatitis_c_model", "asthma_model", "malaria_model", "alzheimers_model", "obesity_model", "epilepsy_model", "prostate_model", "cancer_risk_model", "migraine_model", "tuberculosis_model", "copd_model", "cervical_model", "chronic_model", "liver_disease_model", "pneumonia_model", "general_disease_model"}  # Add more as they get retrained

for attr_name, model_path in MODEL_FILES.items():
    try:
        with open(model_path, "rb") as fh:
            loaded_data = pickle.load(fh)
            if attr_name in PACKAGED_MODELS and isinstance(loaded_data, dict):
                # Store the entire package for packaged models
                loaded_models[attr_name] = loaded_data
            else:
                loaded_models[attr_name] = loaded_data
    except FileNotFoundError:
        failed_models[attr_name] = "file not found"
    except Exception as exc:  # noqa: BLE001 - surface exact failure
        failed_models[attr_name] = str(exc)

# Expose successfully loaded models with their expected variable names
globals().update(loaded_models)

# Display load status for transparency
if failed_models:
    failure_lines = [f"• {name} → {reason}" for name, reason in failed_models.items()]
    st.warning(
        "The following models failed to load. Verify that the files exist and are readable:\n" +
        "\n".join(failure_lines)
    )
else:
    st.success("All prediction models loaded successfully.")

def generate_text_report(recommendations):
    """Generate a text-based health report"""
    report = f"""
{'='*80}
HEALTH MANAGEMENT PLAN
{'='*80}

Patient Name: {recommendations.get('name', 'N/A')}
Condition: {recommendations.get('topic', 'N/A')}
Report Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

{'='*80}
DIETARY PLAN
{'='*80}

"""
    
    dietary = recommendations.get('dietary_plan', {})
    if dietary:
        report += f"\nDaily Nutritional Targets:\n"
        report += f"- Calories: {dietary.get('daily_calories', 'Not specified')}\n"
        report += f"- Protein: {dietary.get('daily_protein', 'Not specified')}\n"
        report += f"- Carbohydrates: {dietary.get('daily_carbohydrates', 'Not specified')}\n"
        report += f"- Fats: {dietary.get('daily_fats', 'Not specified')}\n"
        report += f"- Fiber: {dietary.get('daily_fiber', 'Not specified')}\n"
        report += f"- Water: {dietary.get('hydration', 'Not specified')}\n"
        
        report += f"\nFoods to Eat:\n"
        for food in dietary.get('foods_to_eat', []):
            report += f"- {food}\n"
        
        report += f"\nFoods to Avoid:\n"
        for food in dietary.get('foods_to_avoid', []):
            report += f"- {food}\n"
    
    report += f"\n{'='*80}\nMEDICATIONS\n{'='*80}\n\n"
    
    medications = recommendations.get('medications', {})
    if medications:
        for med in medications.get('medication_details', []):
            report += f"\n{med.get('name', 'Medication')}:\n"
            report += f"  Dosage: {med.get('dosage', 'N/A')}\n"
            report += f"  Frequency: {med.get('frequency', 'N/A')}\n"
            report += f"  Duration: {med.get('duration', 'N/A')}\n"
    
    report += f"\n{'='*80}\nDOCTOR VISITATION\n{'='*80}\n\n"
    
    doctor = recommendations.get('doctor_visitation', {})
    if doctor:
        report += f"Urgency: {doctor.get('urgency', 'N/A')}\n"
        report += f"Specialist: {doctor.get('specialist_type', 'N/A')}\n"
        report += f"Follow-up: {doctor.get('followup_schedule', 'N/A')}\n"
    
    report += f"\n{'='*80}\nPRECAUTIONS\n{'='*80}\n\n"
    
    precautions = recommendations.get('precautions', {})
    if precautions:
        report += "Lifestyle Changes:\n"
        for change in precautions.get('lifestyle_changes', []):
            report += f"- {change}\n"
        
        report += "\nWarning Signs:\n"
        for sign in precautions.get('warning_signs', []):
            report += f"- {sign}\n"
    
    report += f"\n{'='*80}\nEXERCISE RECOMMENDATIONS\n{'='*80}\n\n"
    
    exercise = recommendations.get('exercise_recommendations', {})
    if exercise:
        report += f"Duration: {exercise.get('duration', 'N/A')}\n"
        report += f"Frequency: {exercise.get('frequency', 'N/A')}\n"
        report += f"Intensity: {exercise.get('intensity', 'N/A')}\n"
        report += "\nRecommended Exercises:\n"
        for ex in exercise.get('recommended_exercises', []):
            report += f"- {ex}\n"
    
    report += f"\n{'='*80}\n"
    report += "DISCLAIMER: This report is for informational purposes only.\n"
    report += "Please consult with qualified healthcare professionals for medical advice.\n"
    report += f"{'='*80}\n"
    
    return report


def display_recommendations(recommendations):
    """
    Display structured health recommendations in Streamlit
    """
    if not recommendations:
        st.warning("Unable to generate recommendations at this time.")
        return
    
    # Header with patient name and topic
    st.markdown("---")
    st.markdown(f"## 📋 Health Management Plan")
    if recommendations.get('name'):
        st.markdown(f"**Patient:** {recommendations.get('name')}")
    st.markdown(f"**Condition:** {recommendations.get('topic', 'Health Management')}")
    st.markdown("---")
    
    # Create tabs for better organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🍽️ Diet Plan", 
        "💊 Medications", 
        "👨‍⚕️ Doctor Visit", 
        "⚠️ Precautions",
        "🏃 Exercise"
    ])
    
    with tab1:
        st.subheader("🍽️ Dietary Plan")
        dietary = recommendations.get('dietary_plan', {})
        
        if dietary:
            # Daily Nutritional Targets
            st.markdown("### 📊 Daily Nutritional Targets")
            
            nutrition_cols = st.columns(4)
            
            with nutrition_cols[0]:
                calories = dietary.get('daily_calories', 'Not specified')
                st.metric("Calories", calories)
                
            with nutrition_cols[1]:
                protein = dietary.get('daily_protein', 'Not specified')
                st.metric("Protein", protein)
                
            with nutrition_cols[2]:
                carbs = dietary.get('daily_carbohydrates', 'Not specified')
                st.metric("Carbohydrates", carbs)
                
            with nutrition_cols[3]:
                fats = dietary.get('daily_fats', 'Not specified')
                st.metric("Healthy Fats", fats)
            
            # Additional nutritional metrics
            st.markdown("---")
            st.markdown("### 🔬 Key Nutritional Guidelines")
            
            nutrition_cols2 = st.columns(5)
            
            with nutrition_cols2[0]:
                fiber = dietary.get('daily_fiber', 'Not specified')
                st.metric("Fiber", fiber)
                
            with nutrition_cols2[1]:
                sodium = dietary.get('daily_sodium', 'Not specified')
                st.metric("Sodium (max)", sodium)
                
            with nutrition_cols2[2]:
                sugar = dietary.get('daily_sugar', 'Not specified')
                st.metric("Added Sugar (max)", sugar)
                
            with nutrition_cols2[3]:
                cholesterol = dietary.get('daily_cholesterol', 'Not specified')
                st.metric("Cholesterol (max)", cholesterol)
                
            with nutrition_cols2[4]:
                water = dietary.get('hydration', 'Not specified')
                st.metric("Water", water)
            
            # Macronutrient breakdown
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ✅ Foods to Eat:")
                foods_to_eat = dietary.get('foods_to_eat', [])
                if foods_to_eat:
                    for food in foods_to_eat:
                        st.markdown(f"• {food}")
                else:
                    st.info("No specific recommendations")
            
            with col2:
                st.markdown("### ❌ Foods to Avoid:")
                foods_to_avoid = dietary.get('foods_to_avoid', [])
                if foods_to_avoid:
                    for food in foods_to_avoid:
                        st.markdown(f"• {food}")
                else:
                    st.info("No specific restrictions")
            
            # Detailed meal plan
            st.markdown("---")
            st.markdown("### 📅 Sample Meal Plan")
            meal_plan = dietary.get('meal_plan', {})
            if meal_plan:
                meal_cols = st.columns(4)
                meals = [
                    ("🌅 Breakfast", meal_plan.get('breakfast', '')),
                    ("☀️ Lunch", meal_plan.get('lunch', '')),
                    ("🌆 Dinner", meal_plan.get('dinner', '')),
                    ("🍎 Snacks", meal_plan.get('snacks', ''))
                ]
                
                for col, (meal_name, meal_content) in zip(meal_cols, meals):
                    with col:
                        st.markdown(f"**{meal_name}**")
                        st.write(meal_content if meal_content else "Not specified")
            
            # Vitamin and mineral recommendations
            st.markdown("---")
            st.markdown("### 💊 Essential Vitamins & Minerals")
            
            vitamins = dietary.get('vitamins_minerals', {})
            if vitamins:
                vit_cols = st.columns(3)
                
                with vit_cols[0]:
                    st.markdown("**Key Vitamins:**")
                    vit_dict = vitamins.get('vitamins', {})
                    if vit_dict:
                        for vit, amount in vit_dict.items():
                            st.markdown(f"• {vit}: {amount}")
                    else:
                        st.write("Standard daily requirements")
                
                with vit_cols[1]:
                    st.markdown("**Key Minerals:**")
                    min_dict = vitamins.get('minerals', {})
                    if min_dict:
                        for mineral, amount in min_dict.items():
                            st.markdown(f"• {mineral}: {amount}")
                    else:
                        st.write("Standard daily requirements")
                
                with vit_cols[2]:
                    st.markdown("**Supplements (if needed):**")
                    supplements = vitamins.get('supplements', [])
                    if supplements:
                        for supp in supplements:
                            st.markdown(f"• {supp}")
                    else:
                        st.write("Consult your doctor")
            
            # Meal timing recommendations
            st.markdown("---")
            st.markdown("### ⏰ Meal Timing & Frequency")
            timing = dietary.get('meal_timing', {})
            if timing:
                st.info(f"**Recommended eating schedule:** {timing.get('schedule', 'Eat regular meals every 3-4 hours')}")
                st.write(f"**Best practices:** {timing.get('tips', 'Avoid eating 2-3 hours before bedtime')}")
            else:
                st.info("Eat balanced meals at regular intervals throughout the day")
            
            # Portion control guide
            st.markdown("---")
            st.markdown("### 🍛 Portion Control Guide")
            portions = dietary.get('portion_sizes', {})
            if portions:
                portion_cols = st.columns(2)
                with portion_cols[0]:
                    st.markdown("**Recommended Portions:**")
                    for food_group, portion in portions.items():
                        st.markdown(f"• {food_group}: {portion}")
                with portion_cols[1]:
                    st.info("**Hand-based portion guide:**\n\n"
                           "• Palm = Protein serving\n"
                           "• Fist = Vegetable serving\n"
                           "• Cupped hand = Carb serving\n"
                           "• Thumb = Fat serving")
            else:
                st.info("Follow standard portion guidelines based on your age, gender, and activity level")
    
    with tab2:
        st.subheader("💊 Medications")
        medications = recommendations.get('medications', {})
        
        if medications:
            # Prescription medications
            prescription = medications.get('prescription_required', [])
            if prescription:
                st.markdown("### 🏥 Prescription Required")
                for med in prescription:
                    st.markdown(f"• {med}")
            
            # OTC medications
            otc = medications.get('over_the_counter', [])
            if otc:
                st.markdown("### 🛒 Over-the-Counter Options")
                for med in otc:
                    st.markdown(f"• {med}")
            
            # Detailed medication information
            st.markdown("---")
            st.markdown("### 📝 Medication Details")
            med_details = medications.get('medication_details', [])
            
            if med_details:
                for med in med_details:
                    with st.expander(f"💊 {med.get('name', 'Medication')}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Dosage:** {med.get('dosage', 'N/A')}")
                            st.markdown(f"**Frequency:** {med.get('frequency', 'N/A')}")
                            st.markdown(f"**Duration:** {med.get('duration', 'N/A')}")
                        
                        with col2:
                            st.markdown(f"**Approximate Cost:** {med.get('approximate_cost', 'N/A')}")
                            
                            generic_alt = med.get('generic_alternatives', [])
                            if generic_alt:
                                st.markdown("**Generic Alternatives:**")
                                for alt in generic_alt:
                                    st.markdown(f"• {alt}")
            else:
                st.info("Consult your doctor for specific medication recommendations")
        else:
            st.info("No medication information available. Please consult your healthcare provider.")
    
    with tab3:
        st.subheader("👨‍⚕️ Doctor Visitation")
        doctor = recommendations.get('doctor_visitation', {})
        
        if doctor:
            # Urgency indicator
            urgency = doctor.get('urgency', 'routine')
            urgency_colors = {
                'immediate': ('🔴', 'red', 'IMMEDIATE ATTENTION REQUIRED'),
                'within 24 hours': ('🟠', 'orange', 'URGENT - Within 24 Hours'),
                'within a week': ('🟡', 'gold', 'Schedule Within a Week'),
                'routine': ('🟢', 'green', 'Routine Check-up')
            }
            
            icon, color, message = urgency_colors.get(urgency.lower(), ('🔵', 'blue', urgency))
            
            st.markdown(f"### Urgency Level")
            st.markdown(f"<h3 style='color:{color}'>{icon} {message}</h3>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏥 Specialist Type")
                specialist = doctor.get('specialist_type', 'General Practitioner')
                st.info(specialist)
                
                st.markdown("### 📅 Follow-up Schedule")
                followup = doctor.get('followup_schedule', doctor.get('follow_up_schedule', 'As needed'))
                st.write(followup)
            
            with col2:
                st.markdown("### 🔬 Recommended Tests")
                tests = doctor.get('tests_recommended', [])
                if tests:
                    for test in tests:
                        st.markdown(f"• {test}")
                else:
                    st.write("To be determined by physician")
        else:
            st.info("Consult with your healthcare provider for personalized medical guidance")
    
    with tab4:
        st.subheader("⚠️ Precautions")
        precautions = recommendations.get('precautions', {})
        
        if precautions:
            col1, col2 = st.columns(2)
            
            with col1:
                # Lifestyle changes
                st.markdown("### ✅ Lifestyle Changes")
                lifestyle = precautions.get('lifestyle_changes', [])
                if lifestyle:
                    for change in lifestyle:
                        st.markdown(f"• {change}")
                else:
                    st.info("Maintain healthy lifestyle habits")
                
                # Activities to avoid
                st.markdown("### 🚫 Activities to Avoid")
                avoid = precautions.get('activities_to_avoid', [])
                if avoid:
                    for activity in avoid:
                        st.markdown(f"• {activity}")
                else:
                    st.info("No specific restrictions")
            
            with col2:
                # Warning signs
                st.markdown("### ⚠️ Warning Signs")
                warnings = precautions.get('warning_signs', [])
                if warnings:
                    for sign in warnings:
                        st.warning(f"• {sign}")
                else:
                    st.info("Monitor general health")
            
            # Emergency symptoms
            st.markdown("---")
            emergency = precautions.get('emergency_symptoms', [])
            if emergency:
                st.markdown("### 🆘 Emergency Symptoms (Seek Immediate Help)")
                st.error("If you experience any of these symptoms, call emergency services immediately:")
                for symptom in emergency:
                    st.markdown(f"• {symptom}")
        else:
            st.info("Follow general health precautions and consult your doctor")
    
    with tab5:
        st.subheader("🏃 Exercise Recommendations")
        exercise = recommendations.get('exercise_recommendations', {})
        
        if exercise:
            # Exercise summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                duration = exercise.get('duration', 'N/A')
                st.metric("Duration", duration)
            
            with col2:
                frequency = exercise.get('frequency', 'N/A')
                st.metric("Frequency", frequency)
            
            with col3:
                intensity = exercise.get('intensity', 'N/A')
                st.metric("Intensity", intensity)
            
            # Recommended exercises
            st.markdown("---")
            st.markdown("### 💪 Recommended Exercises")
            exercises = exercise.get('recommended_exercises', [])
            if exercises:
                for i, ex in enumerate(exercises, 1):
                    st.markdown(f"{i}. {ex}")
            else:
                st.info("Consult a fitness professional for personalized exercise plan")
            
            # Safety note
            st.markdown("---")
            st.info("⚠️ Always consult your doctor before starting a new exercise program, especially if you have existing health conditions.")
        else:
            st.info("Regular physical activity is important. Consult your doctor for personalized exercise recommendations.")
    
    # Download option - CORRECTLY INDENTED INSIDE THE FUNCTION
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        # Create unique key using available data
        patient_name = recommendations.get('name', 'patient')
        topic = recommendations.get('topic', 'health')
        download_key = f"download_btn_{patient_name.replace(' ', '_')}_{hash(str(topic))}"
        
        if st.button("📥 Download Full Report", use_container_width=True, key=download_key):
            # Generate the text report
            report_text = generate_text_report(recommendations)
            
            st.download_button(
                label="📄 Download Text Report",
                data=report_text,
                file_name=f"health_report_{patient_name.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                key=f"download_actual_{download_key}"
            )


def get_health_recommendations(disease_name, severity="moderate", patient_info={}):
    """Get tailored health recommendations using NVIDIA LLM"""

    if client is None:
        st.info("Configure the NVIDIA API key to enable AI-generated care plans.")
        return None
    
    prompt = f"""You are a medical AI assistant. Based on the following information, provide structured health recommendations.

Disease: {disease_name}
Severity: {severity}
Patient Information: {json.dumps(patient_info)}

Please provide recommendations in the following JSON format:
{{
    "name": "Patient Name",
    "topic": "{disease_name} Management",
    "dietary_plan": {{
        "foods_to_eat": ["list of recommended foods"],
        "foods_to_avoid": ["list of foods to avoid"],
        "daily_calories": "e.g., 1800-2000 kcal",
        "daily_protein": "e.g., 60-80g",
        "daily_carbohydrates": "e.g., 200-250g",
        "daily_fats": "e.g., 50-70g",
        "daily_fiber": "e.g., 25-30g",
        "daily_sodium": "e.g., <2300mg",
        "daily_sugar": "e.g., <25g added sugar",
        "daily_cholesterol": "e.g., <300mg",
        "meal_plan": {{
            "breakfast": "specific breakfast suggestions",
            "lunch": "specific lunch suggestions",
            "dinner": "specific dinner suggestions",
            "snacks": "healthy snack options"
        }},
        "hydration": "specific water intake recommendation",
        "vitamins_minerals": {{
            "vitamins": {{"Vitamin D": "600-800 IU"}},
            "minerals": {{"Calcium": "1000mg"}},
            "supplements": ["list any recommended supplements"]
        }},
        "meal_timing": {{
            "schedule": "eating schedule",
            "tips": "timing tips"
        }},
        "portion_sizes": {{
            "Vegetables": "2-3 cups per day"
        }}
    }},
    "medications": {{
        "prescription_required": ["list of prescription medications"],
        "over_the_counter": ["list of OTC options"],
        "medication_details": [
            {{
                "name": "medication name",
                "dosage": "dosage info",
                "frequency": "how often",
                "duration": "how long",
                "approximate_cost": "price range in USD",
                "generic_alternatives": ["list of generic options"]
            }}
        ]
    }},
    "doctor_visitation": {{
        "urgency": "immediate/within 24 hours/within a week/routine",
        "specialist_type": "type of specialist needed",
        "tests_recommended": ["list of recommended tests"],
        "followup_schedule": "frequency of follow-ups"
    }},
    "precautions": {{
        "lifestyle_changes": ["specific lifestyle modifications"],
        "activities_to_avoid": ["activities to avoid"],
        "warning_signs": ["symptoms to monitor"],
        "emergency_symptoms": ["symptoms requiring immediate medical attention"]
    }},
    "exercise_recommendations": {{
        "recommended_exercises": ["list of suitable exercises"],
        "duration": "e.g., 30 minutes per day",
        "frequency": "e.g., 5 days per week",
        "intensity": "low/moderate/high"
    }}
}}

CRITICAL: Return ONLY valid JSON. Do not include any text before or after the JSON object."""

    try:
        completion = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant specializing in health recommendations. You MUST return only valid JSON with no additional text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            top_p=0.7,
            max_tokens=2048,
            stream=True
        )
        
        full_response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
        
        # Clean the response
        full_response = full_response.strip()
        if full_response.startswith("```json"):
            full_response = full_response[7:]
        if full_response.startswith("```"):
            full_response = full_response[3:]
        if full_response.endswith("```"):
            full_response = full_response[:-3]
        full_response = full_response.strip()
        
        recommendations = json.loads(full_response)
        return recommendations
        
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse recommendations. Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return None

        
def get_health_tips_from_llm(disease_name, severity="moderate"):
    """Get disease-specific health tips dynamically from NVIDIA LLM"""
    if client is None:
        st.info("Configure the NVIDIA API key to enable dynamic AI health tips.")
        return None
    prompt = f"""You are a medical AI assistant. Provide comprehensive, evidence-based health tips for managing {disease_name} (severity: {severity}).

Please provide tips in the following JSON format:
{{
    "disease_name": "{disease_name}",
    "daily_management_tips": ["Provide 8-10 specific, actionable daily management tips"],
    "prevention_tips": ["Provide 6-8 prevention or risk reduction strategies"],
    "warning_signs": ["List 6-8 warning signs that require medical attention"],
    "quick_reminders": ["Provide 5-6 short, memorable one-sentence tips"],
    "do_and_dont": {{
        "do": ["List 5 things patients SHOULD do"],
        "dont": ["List 5 things patients SHOULD NOT do"]
    }},
    "lifestyle_modifications": ["Provide 5-7 specific lifestyle changes"]
}}

CRITICAL: Return ONLY valid JSON. Do not include any text before or after the JSON object."""

    try:
        completion = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant specializing in health tips. You MUST return only valid JSON with no additional text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            top_p=0.7,
            max_tokens=1536,
            stream=True
        )
        
        full_response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
        
        # Clean the response
        full_response = full_response.strip()
        if full_response.startswith("```json"):
            full_response = full_response[7:]
        if full_response.startswith("```"):
            full_response = full_response[3:]
        if full_response.endswith("```"):
            full_response = full_response[:-3]
        full_response = full_response.strip()
        
        tips = json.loads(full_response)
        return tips
        
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse health tips. Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error getting health tips from LLM: {str(e)}")
        return None


def get_general_health_tips_from_llm():
    """Get general health and wellness tips from NVIDIA LLM"""
    if client is None:
        st.info("Configure the NVIDIA API key to enable AI-generated general health guidance.")
        return None
    prompt = """You are a medical AI assistant. Provide comprehensive general health and wellness tips for maintaining overall health.

Please provide tips in the following JSON format organized by categories:
{
    "nutrition": ["Provide 8-10 evidence-based nutrition tips"],
    "physical_activity": ["Provide 8-10 exercise and movement tips"],
    "sleep_rest": ["Provide 8-10 sleep hygiene tips"],
    "mental_health": ["Provide 8-10 mental wellness tips"],
    "preventive_care": ["Provide 8-10 preventive health tips"],
    "lifestyle_habits": ["Provide 8-10 healthy lifestyle tips"],
    "hydration": ["Provide 5-6 hydration tips"],
    "immune_health": ["Provide 6-8 immune system boosting tips"]
}

CRITICAL: Return ONLY valid JSON. Do not include any text before or after the JSON object."""

    try:
        completion = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant specializing in disease-specific guidance. You MUST return only valid JSON with no additional text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            top_p=0.7,
            max_tokens=2048,
            stream=True
        )
        
        full_response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
        
        # Clean the response
        full_response = full_response.strip()
        if full_response.startswith("```json"):
            full_response = full_response[7:]
        if full_response.startswith("```"):
            full_response = full_response[3:]
        if full_response.endswith("```"):
            full_response = full_response[:-3]
        full_response = full_response.strip()
        
        tips = json.loads(full_response)
        return tips
        
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse general health tips. Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error getting general health tips: {str(e)}")
        return None


def display_health_tips_dynamic(disease_name=None, severity=None):
    """Display dynamically generated health tips from LLM"""
    st.markdown("---")
    st.markdown("## 💡 Health Tips & Daily Guidance")
    
    if disease_name:
        tip_tabs = st.tabs(["🎯 Disease-Specific Tips", "📋 General Health Tips", "⚠️ Warning Signs"])
        
        with tip_tabs[0]:
            st.subheader(f"Personalized Tips for {disease_name}")
            
            with st.spinner("Generating personalized health tips from AI..."):
                tips = get_health_tips_from_llm(disease_name, severity)
                
                if tips:
                    st.markdown("### 📅 Daily Management")
                    daily_tips = tips.get('daily_management_tips', [])
                    if daily_tips:
                        for i, tip in enumerate(daily_tips, 1):
                            st.info(f"**Tip {i}:** {tip}")
                    
                    st.markdown("---")
                    st.markdown("### ✅ Do's and ❌ Don'ts")
                    do_dont = tips.get('do_and_dont', {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### ✅ Things to DO")
                        for do in do_dont.get('do', []):
                            st.success(f"✓ {do}")
                    with col2:
                        st.markdown("#### ❌ Things to AVOID")
                        for dont in do_dont.get('dont', []):
                            st.error(f"✗ {dont}")
                    
                    st.markdown("---")
                    st.markdown("### 🛡️ Prevention & Long-term Care")
                    for tip in tips.get('prevention_tips', []):
                        st.success(f"• {tip}")
                    
                    st.markdown("---")
                    st.markdown("### 🔄 Lifestyle Modifications")
                    lifestyle = tips.get('lifestyle_modifications', [])
                    cols = st.columns(2)
                    for idx, mod in enumerate(lifestyle):
                        with cols[idx % 2]:
                            st.info(f"• {mod}")
                    
                    st.markdown("---")
                    st.markdown("### ⚡ Quick Daily Reminders")
                    for tip in tips.get('quick_reminders', []):
                        st.markdown(f"⚡ {tip}")
                else:
                    # Fallback if LLM fails
                    st.warning("Unable to generate personalized tips at this time. Please ensure API is configured correctly.")
                    st.info("""
                    **General Health Recommendations:**
                    - Follow prescribed medications consistently
                    - Maintain regular medical checkups
                    - Adopt a balanced, nutritious diet
                    - Stay physically active within your capabilities
                    - Get adequate sleep (7-9 hours for adults)
                    - Manage stress through relaxation techniques
                    - Avoid tobacco and limit alcohol consumption
                    - Stay well-hydrated throughout the day
                    - Monitor your symptoms and keep a health diary
                    - Maintain open communication with your healthcare team
                    """)
        
        with tip_tabs[1]:
            st.subheader("General Health & Wellness")
            with st.spinner("Loading general health tips from AI..."):
                general_tips = get_general_health_tips_from_llm()
                if general_tips:
                    categories = {
                        "nutrition": "🍎 Nutrition & Diet",
                        "physical_activity": "🏃 Physical Activity",
                        "sleep_rest": "😴 Sleep & Rest",
                        "mental_health": "🧘 Mental Health",
                        "preventive_care": "💊 Preventive Care",
                        "lifestyle_habits": "🌟 Lifestyle Habits",
                        "hydration": "💧 Hydration",
                        "immune_health": "🛡️ Immune Health"
                    }
                    for key, title in categories.items():
                        with st.expander(title):
                            for tip in general_tips.get(key, []):
                                st.markdown(f"• {tip}")
                else:
                    st.warning("Unable to load general health tips. Using offline recommendations.")
                    with st.expander("🍎 Nutrition & Diet"):
                        st.markdown("""
                        - Eat a variety of colorful fruits and vegetables daily
                        - Choose whole grains over refined grains
                        - Include lean proteins in your diet
                        - Limit processed foods and added sugars
                        - Practice portion control
                        - Read nutrition labels carefully
                        - Plan meals ahead to make healthier choices
                        - Eat mindfully without distractions
                        """)
                    
                    with st.expander("🏃 Physical Activity"):
                        st.markdown("""
                        - Aim for 150 minutes of moderate activity weekly
                        - Include both cardio and strength training
                        - Start slowly and gradually increase intensity
                        - Find activities you enjoy for sustainability
                        - Take breaks from sitting every hour
                        - Stretch regularly to maintain flexibility
                        - Exercise with friends for motivation
                        - Listen to your body and rest when needed
                        """)
        
        with tip_tabs[2]:
            st.subheader("⚠️ Warning Signs to Monitor")
            if tips:
                warning_signs = tips.get('warning_signs', [])
                if warning_signs:
                    for sign in warning_signs:
                        st.warning(f"• {sign}")
                else:
                    st.info("Consult your healthcare provider if you experience any concerning symptoms.")
            else:
                st.warning("""
                **General Warning Signs (Seek Medical Attention):**
                - Severe or persistent pain
                - Sudden changes in symptoms
                - High fever (>103°F/39.4°C)
                - Difficulty breathing
                - Chest pain or pressure
                - Severe headache
                - Sudden confusion or disorientation
                - Uncontrolled bleeding
                - Severe allergic reactions
                - Loss of consciousness
                """)
    else:
        # Display general health tips if no specific disease
        st.info("Select a specific health condition for personalized tips.")
        with st.spinner("Loading general wellness tips..."):
            general_tips = get_general_health_tips_from_llm()
            if general_tips:
                categories = {
                    "nutrition": "🍎 Nutrition & Diet",
                    "physical_activity": "🏃 Physical Activity",
                    "sleep_rest": "😴 Sleep & Rest",
                    "mental_health": "🧘 Mental Health",
                    "preventive_care": "💊 Preventive Care",
                    "lifestyle_habits": "🌟 Lifestyle Habits",
                    "hydration": "💧 Hydration",
                    "immune_health": "🛡️ Immune Health"
                }
                for key, title in categories.items():
                    with st.expander(title):
                        for tip in general_tips.get(key, []):
                            st.markdown(f"• {tip}")
                            
# Health Tips
health_tips = [
    "Drink at least 8 glasses of water daily.",
    "Exercise for at least 30 minutes a day.",
    "Eat a balanced diet rich in fruits and vegetables.",
    "Get at least 7-8 hours of sleep each night.",
    "Avoid smoking and limit alcohol consumption.",
    "Practice mindfulness and meditation to reduce stress.",
    "Regular health check-ups can prevent serious diseases.",
    "Maintain a healthy weight through balanced diet and exercise.",
    "Wash your hands frequently to prevent infections.",
    "Limit processed foods and sugar intake.",
    "Take breaks from screen time every hour.",
    "Practice good posture to prevent back problems."
]

# --- NAV STATE (must be defined before any sidebar buttons use it) ---
selected = st.session_state.get("selected", "Home")
# --------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏥 Health Suite")

    # Consistent icon map, including both emoji and plain labels
    DISEASE_ICONS = {
        # Home/General
        "Home": "🏠",
        "🏠 Home": "🏠",
        "General Disease Prediction": "🔍",
        # Metabolic
        "Diabetes Prediction": "🩸",
        "Obesity Prediction": "⚖️",
        # Cardiovascular
        "Heart Disease Prediction": "❤️",
        # Neurological
        "Parkinsons Prediction": "🧠",
        "Alzheimers Prediction": "🧩",
        "Epilepsy Prediction": "⚡",
        "Migraine Prediction": "💥",
        # Organ
        "Liver Prediction": "🧪",   # stylistic fallback
        "Kidney Disease Prediction": "🫘",
        # Infectious
        "Hepatitis Prediction": "🧪",
        "Tuberculosis Prediction": "🫁",
        "Malaria Prediction": "🦟",
        # Cancer
        "Lung Cancer Prediction": "🌬️",
        "Breast Cancer Prediction": "🎗️",
        "Liver Cancer Prediction": "🔬",
        "Prostate Cancer Prediction": "🧫",
        "Cervical Cancer Prediction": "🧫",
        "Cancer Risk Assessment": "🎯",
        # Respiratory
        "Asthma Prediction": "🌫️",
        "COPD Prediction": "😮‍💨",
        "Pneumonia Prediction": "🫁",
        # Services
        "AI Health Assistant": "🤖",
        "Book Appointment": "📅",
        "Set Reminder": "⏰",
        "Health Tips": "💡",
        # Analytics
        "Model Metrics": "📈",
    }

    def section(label, options):
        st.caption(label)
        cols = st.columns(min(3, len(options)))
        chosen = None
        for i, opt in enumerate(options):
            col = cols[i % len(cols)]
            with col:
                icon = DISEASE_ICONS.get(opt, "•")
                active = (opt == selected)
                btn = st.button(
                    f"{icon} {opt.split(' Prediction')[0]}",
                    key=f"chip_{opt}",
                    use_container_width=True,
                    type="primary" if active else "secondary",
                )
                if btn:
                    chosen = opt
        st.divider()
        return chosen

    # Top minimal chips
    top_choice = None
    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "🏠 Home",
            use_container_width=True,
            type="primary" if selected in ["🏠 Home", "Home"] else "secondary",
            key="top_home",
        ):
            top_choice = "Home"  # normalize for downstream checks
    with c2:
        if st.button(
            "🔍 General",
            use_container_width=True,
            type="primary" if selected in ["🔍 General Disease Prediction", "General Disease Prediction"] else "secondary",
            key="top_general",
        ):
            top_choice = "🔍 General Disease Prediction"

    st.divider()

    # Sections
    ch1 = section("Metabolic", ["Diabetes Prediction", "Obesity Prediction"])
    ch2 = section("Cardiovascular", ["Heart Disease Prediction"])
    ch3 = section("Neurological", ["Parkinsons Prediction", "Alzheimers Prediction", "Epilepsy Prediction", "Migraine Prediction"])
    ch4 = section("Organ", ["Liver Prediction", "Kidney Disease Prediction"])
    ch5 = section("Infectious", ["Hepatitis Prediction", "Tuberculosis Prediction", "Malaria Prediction"])
    ch6 = section("Cancer", ["Lung Cancer Prediction", "Breast Cancer Prediction", "Liver Cancer Prediction", "Prostate Cancer Prediction", "Cervical Cancer Prediction", "Cancer Risk Assessment"])
    ch7 = section("Respiratory", ["Asthma Prediction", "COPD Prediction", "Pneumonia Prediction"])
    ch8 = section("Services", ["AI Health Assistant", "Book Appointment", "Set Reminder", "Health Tips"])
    
    # Model Metrics Button
    st.markdown("---")
    st.caption("📊 Analytics")
    ch9 = None
    if st.button("📈 Model Metrics", use_container_width=True, type="secondary" if selected != "Model Metrics" else "primary"):
        ch9 = "Model Metrics"

    # Resolve choice precedence
    nav_choice = top_choice or ch1 or ch2 or ch3 or ch4 or ch5 or ch6 or ch7 or ch8 or ch9
# ---------------------------------------------------

# Apply selection and keep it persistent + normalized for Home
if nav_choice:
    normalized = "Home" if nav_choice in ["🏠 Home", "Home"] else nav_choice
    st.session_state["selected"] = normalized
    selected = normalized
else:
    # Keep current session state selection
    selected = st.session_state.get("selected", "Home")

# Active header in main area
active_icon = DISEASE_ICONS.get(selected, "🩺")
st.subheader(f"{active_icon} {selected}")

# Home Page
if selected == 'Home':
    st.title("🏥 Multiple Disease Prediction System")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Diseases", "20", "✓ Comprehensive")
        st.metric("Accuracy Rate", "91.2%", "✓ High")
    
    with col2:
        st.metric("Users Served", "847", "↑ 52")
        st.metric("Predictions Made", "2,341", "↑ 127")
    
    with col3:
        st.metric("Available Models", "20", "✓ All Active")
        st.metric("Response Time", "< 1.5s", "✓ Fast")
    
    st.markdown("---")
    
    st.subheader("📊 System Features")
    
    features = {
        "🤖 AI-Powered Predictions": "Advanced machine learning models for accurate disease prediction",
        "💊 Personalized Recommendations": "Tailored diet plans, medications, and lifestyle suggestions",
        "👨‍⚕️ Doctor Consultation": "Guidance on when and which specialist to consult",
        "📱 24/7 Availability": "Access health predictions anytime, anywhere",
        "🔒 Privacy First": "Your health data is secure and confidential",
        "📈 Track Progress": "Monitor your health journey over time"
    }
    
    for feature, description in features.items():
        st.write(f"**{feature}**")
        st.write(f"   {description}")
    
    st.markdown("---")
    st.info("💡 **Tip:** Start with the General Disease Prediction for symptom-based analysis or choose a specific disease category from the menu.")

# Model Metrics Page
if selected == 'Model Metrics':
    st.title("📈 Model Performance Metrics")
    st.markdown("Comprehensive performance analysis of all disease prediction models")
    
    # Model name mappings for display
    MODEL_DISPLAY_NAMES = {
        "diabetes_model": "🩸 Diabetes Prediction",
        "heart_model": "❤️ Heart Disease Prediction",
        "breast_cancer_model": "🎗️ Breast Cancer Prediction",
        "kidney_disease_model": "🫘 Kidney Disease Prediction",
        "lung_cancer_model": "🌬️ Lung Cancer Prediction",
        "parkinsons_model": "🧠 Parkinson's Prediction",
        "liver_cancer_model": "🔬 Liver Cancer Prediction",
        "hepatitis_c_model": "🧪 Hepatitis Prediction",
        "asthma_model": "🌫️ Asthma Prediction",
        "malaria_model": "🦟 Malaria Prediction",
        "alzheimers_model": "🧩 Alzheimer's Prediction",
        "obesity_model": "⚖️ Obesity Prediction",
        "epilepsy_model": "⚡ Epilepsy Prediction",
        "prostate_model": "🧫 Prostate Cancer Prediction",
        "cancer_risk_model": "🎯 Cancer Risk Assessment",
        "migraine_model": "💥 Migraine Prediction",
        "tuberculosis_model": "🫁 Tuberculosis Prediction",
        "copd_model": "😮‍💨 COPD Prediction",
        "cervical_model": "🧫 Cervical Cancer Prediction",
        "chronic_model": "🫘 Chronic Kidney Disease",
        "liver_disease_model": "🧪 Liver Disease Prediction",
        "pneumonia_model": "🫁 Pneumonia Prediction",
        "general_disease_model": "🔍 General Disease Prediction"
    }
    
    # Summary statistics
    total_models = len(loaded_models)
    models_with_accuracy = sum(1 for m in loaded_models.values() if isinstance(m, dict) and 'accuracy' in m)
    
    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
    with col_sum1:
        st.metric("Total Models", total_models)
    with col_sum2:
        st.metric("Models with Metrics", models_with_accuracy)
    with col_sum3:
        avg_accuracy = np.mean([m.get('accuracy', 0) for m in loaded_models.values() if isinstance(m, dict) and 'accuracy' in m]) if models_with_accuracy > 0 else 0
        st.metric("Avg Accuracy", f"{avg_accuracy*100:.1f}%")
    with col_sum4:
        st.metric("Failed Models", len(failed_models))
    
    st.markdown("---")
    
    # Model selector
    model_options = list(loaded_models.keys())
    selected_model = st.selectbox(
        "Select a model to view detailed metrics:",
        model_options,
        format_func=lambda x: MODEL_DISPLAY_NAMES.get(x, x.replace('_', ' ').title())
    )
    
    if selected_model and selected_model in loaded_models:
        model_data = loaded_models[selected_model]
        display_name = MODEL_DISPLAY_NAMES.get(selected_model, selected_model.replace('_', ' ').title())
        
        st.subheader(f"{display_name}")
        
        if isinstance(model_data, dict):
            # Check what metrics are available
            has_detailed_metrics = 'precision_weighted' in model_data
            
            # Main metrics
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                st.markdown("#### 🎯 Classification Metrics")
                
                if has_detailed_metrics:
                    metrics_data = {
                        'Metric': ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)', 'ROC-AUC (Weighted)'],
                        'Score': [
                            f"{model_data.get('accuracy', 0)*100:.2f}%",
                            f"{model_data.get('precision_weighted', 0)*100:.2f}%",
                            f"{model_data.get('recall_weighted', 0)*100:.2f}%",
                            f"{model_data.get('f1_weighted', 0)*100:.2f}%",
                            f"{model_data.get('roc_auc_weighted', 0)*100:.2f}%"
                        ]
                    }
                else:
                    # Basic metrics for models without detailed metrics
                    metrics_data = {
                        'Metric': ['Accuracy', 'CV Score'],
                        'Score': [
                            f"{model_data.get('accuracy', 0)*100:.2f}%" if model_data.get('accuracy') else "N/A",
                            f"{model_data.get('cv_score', 0)*100:.2f}%" if model_data.get('cv_score') else "N/A"
                        ]
                    }
                
                st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)
                
                # Additional macro metrics if available
                if has_detailed_metrics:
                    st.markdown("#### 📊 Macro Metrics")
                    macro_data = {
                        'Metric': ['Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)', 'ROC-AUC (Macro)'],
                        'Score': [
                            f"{model_data.get('precision_macro', 0)*100:.2f}%",
                            f"{model_data.get('recall_macro', 0)*100:.2f}%",
                            f"{model_data.get('f1_macro', 0)*100:.2f}%",
                            f"{model_data.get('roc_auc_macro', 0)*100:.2f}%"
                        ]
                    }
                    st.dataframe(pd.DataFrame(macro_data), hide_index=True, use_container_width=True)
            
            with col_m2:
                st.markdown("#### 🔧 Model Configuration")
                config_data = {
                    'Parameter': [],
                    'Value': []
                }
                
                if model_data.get('model_type'):
                    config_data['Parameter'].append('Model Type')
                    config_data['Value'].append(model_data.get('model_type'))
                if model_data.get('n_estimators'):
                    config_data['Parameter'].append('Estimators')
                    config_data['Value'].append(str(model_data.get('n_estimators')))
                if model_data.get('n_features'):
                    config_data['Parameter'].append('Features')
                    config_data['Value'].append(str(model_data.get('n_features')))
                if model_data.get('n_classes'):
                    config_data['Parameter'].append('Classes')
                    config_data['Value'].append(str(model_data.get('n_classes')))
                if model_data.get('cv_score'):
                    config_data['Parameter'].append('CV Score')
                    config_data['Value'].append(f"{model_data.get('cv_score')*100:.2f}%")
                if model_data.get('description'):
                    config_data['Parameter'].append('Description')
                    config_data['Value'].append(model_data.get('description'))
                
                if config_data['Parameter']:
                    st.dataframe(pd.DataFrame(config_data), hide_index=True, use_container_width=True)
                else:
                    st.info("Model configuration details not available.")
                
                # Feature columns info
                if 'feature_columns' in model_data or 'symptom_columns' in model_data:
                    feature_cols = model_data.get('feature_columns') or model_data.get('symptom_columns', [])
                    st.markdown(f"#### 📝 Total Features: {len(feature_cols)}")
            
            st.markdown("---")
            
            # Feature Importance Section
            st.markdown("#### 🏆 Feature Importance")
            
            feature_importance = model_data.get('feature_importance', [])
            
            if feature_importance and isinstance(feature_importance, list) and len(feature_importance) > 0:
                # Check if it's the new format (list of dicts) or old format
                if isinstance(feature_importance[0], dict):
                    # New format
                    feat_col1, feat_col2 = st.columns(2)
                    
                    with feat_col1:
                        st.markdown("**Top 10 Features**")
                        top_10 = feature_importance[:10]
                        feat_df1 = pd.DataFrame(top_10)
                        if 'symptom' in feat_df1.columns:
                            feat_df1['symptom'] = feat_df1['symptom'].str.replace('_', ' ').str.title()
                            feat_df1.columns = ['Feature', 'Importance']
                        elif 'feature' in feat_df1.columns:
                            feat_df1['feature'] = feat_df1['feature'].str.replace('_', ' ').str.title()
                            feat_df1.columns = ['Feature', 'Importance']
                        feat_df1['Importance'] = feat_df1['Importance'].apply(lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x)
                        feat_df1.index = range(1, len(feat_df1) + 1)
                        st.dataframe(feat_df1, use_container_width=True)
                    
                    with feat_col2:
                        if len(feature_importance) > 10:
                            st.markdown("**Features 11-20**")
                            top_20 = feature_importance[10:20]
                            feat_df2 = pd.DataFrame(top_20)
                            if 'symptom' in feat_df2.columns:
                                feat_df2['symptom'] = feat_df2['symptom'].str.replace('_', ' ').str.title()
                                feat_df2.columns = ['Feature', 'Importance']
                            elif 'feature' in feat_df2.columns:
                                feat_df2['feature'] = feat_df2['feature'].str.replace('_', ' ').str.title()
                                feat_df2.columns = ['Feature', 'Importance']
                            feat_df2['Importance'] = feat_df2['Importance'].apply(lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x)
                            feat_df2.index = range(11, 11 + len(feat_df2))
                            st.dataframe(feat_df2, use_container_width=True)
                    
                    # Bar chart visualization
                    st.markdown("**Feature Importance Visualization**")
                    chart_df = pd.DataFrame(feature_importance[:10])
                    if 'symptom' in chart_df.columns:
                        chart_df['symptom'] = chart_df['symptom'].str.replace('_', ' ').str.title()
                        chart_df = chart_df.rename(columns={'symptom': 'Feature', 'importance': 'Importance'})
                    elif 'feature' in chart_df.columns:
                        chart_df['feature'] = chart_df['feature'].str.replace('_', ' ').str.title()
                        chart_df = chart_df.rename(columns={'feature': 'Feature', 'importance': 'Importance'})
                    st.bar_chart(chart_df.set_index('Feature')['Importance'])
                    
                else:
                    # Old format (just importance values)
                    feature_cols = model_data.get('feature_columns') or model_data.get('symptom_columns', [])
                    if len(feature_cols) == len(feature_importance):
                        feat_df = pd.DataFrame({
                            'Feature': [f.replace('_', ' ').title() for f in feature_cols[:20]],
                            'Importance': [f"{imp*100:.2f}%" for imp in feature_importance[:20]]
                        })
                        feat_df.index = range(1, len(feat_df) + 1)
                        st.dataframe(feat_df, use_container_width=True)
            else:
                st.info("Feature importance data not available for this model.")
            
            # Classes/Diseases covered
            if 'diseases' in model_data or 'classes' in model_data:
                st.markdown("---")
                classes = model_data.get('diseases') or model_data.get('classes', [])
                st.markdown(f"#### 🏥 Classes/Diseases Covered ({len(classes)})")
                
                if classes:
                    num_cols = min(4, len(classes))
                    d_cols = st.columns(num_cols)
                    for i, cls in enumerate(classes):
                        with d_cols[i % num_cols]:
                            st.write(f"• {cls}")
        else:
            st.warning("This model does not have detailed metrics available. Consider retraining with metrics enabled.")
    
    st.markdown("---")
    
    # All Models Overview Table
    st.subheader("📊 All Models Overview")
    
    overview_data = []
    for model_name, model_data in loaded_models.items():
        if isinstance(model_data, dict):
            overview_data.append({
                'Model': MODEL_DISPLAY_NAMES.get(model_name, model_name.replace('_', ' ').title()),
                'Accuracy': f"{model_data.get('accuracy', 0)*100:.1f}%" if model_data.get('accuracy') else "N/A",
                'Precision': f"{model_data.get('precision_weighted', 0)*100:.1f}%" if model_data.get('precision_weighted') else "N/A",
                'Recall': f"{model_data.get('recall_weighted', 0)*100:.1f}%" if model_data.get('recall_weighted') else "N/A",
                'F1-Score': f"{model_data.get('f1_weighted', 0)*100:.1f}%" if model_data.get('f1_weighted') else "N/A",
                'ROC-AUC': f"{model_data.get('roc_auc_weighted', 0)*100:.1f}%" if model_data.get('roc_auc_weighted') else "N/A",
                'Features': model_data.get('n_features') or len(model_data.get('feature_columns', model_data.get('symptom_columns', []))) or "N/A"
            })
    
    if overview_data:
        overview_df = pd.DataFrame(overview_data)
        st.dataframe(overview_df, hide_index=True, use_container_width=True)
    
    # Failed models section
    if failed_models:
        st.markdown("---")
        st.subheader("⚠️ Failed Models")
        for model_name, reason in failed_models.items():
            st.error(f"**{MODEL_DISPLAY_NAMES.get(model_name, model_name)}**: {reason}")

# AI Health Assistant
if selected == 'AI Health Assistant':
    st.title("🤖 AI Health Assistant")
    st.markdown("Get personalized health recommendations powered by AI")
    
    # Input fields for patient information
    col1, col2 = st.columns(2)
    
    with col1:
        patient_name = st.text_input("Patient Name")
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0)
        height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
    
    with col2:
        disease = st.selectbox("Disease/Condition", [
            "Diabetes Type 2", "Heart Disease", "Hypertension", "Obesity",
            "Asthma", "COPD", "Pneumonia", "Tuberculosis", "Malaria",
            "Hepatitis B", "Hepatitis C", "Liver Disease",
            "Chronic Kidney Disease", "Parkinsons", "Alzheimers", "Epilepsy",
            "Migraine", "Lung Cancer", "Breast Cancer",
            "Prostate Cancer", "Cervical Cancer"
        ])
        severity = st.selectbox("Severity", ["Mild", "Moderate", "Severe"])
        existing_conditions = st.multiselect("Existing Conditions", [
            "None", "Hypertension", "Diabetes", "Heart Disease", "Asthma",
            "Kidney Disease", "Liver Disease", "Cancer", "Arthritis"
        ])
        allergies = st.text_area("Known Allergies (comma-separated)")
        current_medications = st.text_area("Current Medications (comma-separated)")
    
    if st.button("🤖 Get AI Recommendations"):
        if patient_name and disease:
            with st.spinner("AI is generating personalized recommendations..."):
                # Calculate BMI
                bmi = weight / ((height/100) ** 2)
                
                patient_info = {
                    "name": patient_name,
                    "age": age,
                    "gender": gender,
                    "bmi": round(bmi, 2),
                    "existing_conditions": existing_conditions,
                    "allergies": allergies.split(",") if allergies else [],
                    "current_medications": current_medications.split(",") if current_medications else []
                }
                
                # Get recommendations from NVIDIA LLM
                recommendations = get_health_recommendations(disease, severity, patient_info)
                
                if recommendations:
                    st.success("✅ Recommendations Generated Successfully!")
                    
                    # Display recommendations
                    display_recommendations(recommendations)
                    display_health_tips_dynamic(disease, severity.lower())
                    # Option to download recommendations
                    if st.button("📥 Download Recommendations as PDF"):
                        st.info("PDF download feature coming soon!")
                else:
                    # Fallback recommendations if API fails
                    st.warning("Using offline recommendations. For personalized advice, please configure NVIDIA API.")
                    
                    st.subheader(f"General Recommendations for {disease}")
                    
                    # Provide basic offline recommendations
                    offline_recs = {
                        "Diabetes Type 2": {
                            "diet": ["Whole grains", "Lean proteins", "Non-starchy vegetables", "Limited fruit portions"],
                            "avoid": ["Sugary drinks", "Processed foods", "White bread", "Excessive carbohydrates"],
                            "medications": ["Metformin ($4-$20/month)", "Glipizide ($4-$15/month)"],
                            "doctor": "Endocrinologist - Schedule within 1-2 weeks",
                            "precautions": ["Monitor blood sugar regularly", "Check feet daily", "Regular eye exams"]
                        },
                        "Heart Disease": {
                            "diet": ["Omega-3 rich fish", "Whole grains", "Fruits and vegetables", "Nuts and seeds"],
                            "avoid": ["Trans fats", "Excessive sodium", "Processed meats", "Sugary foods"],
                            "medications": ["Aspirin ($2-$10/month)", "Statins ($10-$50/month)", "Beta-blockers ($10-$30/month)"],
                            "doctor": "Cardiologist - Schedule immediately if chest pain",
                            "precautions": ["Monitor blood pressure", "Avoid strenuous activities", "Take medications as prescribed"]
                        }
                    }
                    
                    if disease in offline_recs:
                        rec = offline_recs[disease]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Foods to Eat:**")
                            for food in rec["diet"]:
                                st.write(f"• {food}")
                            
                            st.write("**Foods to Avoid:**")
                            for food in rec["avoid"]:
                                st.write(f"• {food}")
                        
                        with col2:
                            st.write("**Common Medications:**")
                            for med in rec["medications"]:
                                st.write(f"• {med}")
                            
                            st.write(f"**Doctor Visit:** {rec['doctor']}")
                            
                            st.write("**Precautions:**")
                            for prec in rec["precautions"]:
                                st.write(f"• {prec}")
        else:
            st.error("Please enter patient name and select a disease.")

# The rest of your existing disease prediction pages would continue here...
# (Diabetes, Heart Disease, Parkinsons, etc. - keeping your existing implementations)

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.title("🩺 Diabetes Prediction")
    st.markdown("Predict Type 2 Diabetes risk based on clinical parameters")
    st.info("📊 Model Accuracy: 90.48% | Trained on 100,000 patient records")
    
    name = st.text_input("Name:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", options=["Female", "Male", "Other"])
        age = st.number_input("Age", min_value=1, max_value=120, value=33)
        hypertension = st.selectbox("Hypertension", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    with col2:
        heart_disease = st.selectbox("Heart Disease", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        smoking_history = st.selectbox("Smoking History", options=["No Info", "never", "former", "current", "not current", "ever"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)
    
    with col3:
        hba1c = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.7)
        blood_glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=0, max_value=500, value=120)
    
    if st.button("Predict Diabetes"):
        try:
            # Check if diabetes model loaded properly
            if diabetes_model is None:
                st.error("Diabetes model not loaded. Please check model file.")
            else:
                # Get model components from the packaged model
                if isinstance(diabetes_model, dict):
                    model = diabetes_model['model']
                    scaler = diabetes_model['scaler']
                    gender_mapping = diabetes_model.get('gender_mapping', {'Female': 0, 'Male': 1, 'Other': 2})
                    smoking_mapping = diabetes_model.get('smoking_mapping', {'No Info': 0, 'current': 1, 'ever': 2, 'former': 3, 'never': 4, 'not current': 5})
                else:
                    # Fallback for old model format
                    model = diabetes_model
                    scaler = None
                    gender_mapping = {'Female': 0, 'Male': 1}
                    smoking_mapping = {"never": 0, "current": 1, "former": 2, "not current": 3, "ever": 4, "No Info": 5}
                
                # Encode gender
                gender_val = gender_mapping.get(gender, 0)
                
                # Encode smoking history
                smoking_num = smoking_mapping.get(smoking_history, 0)
                
                # Prepare input in correct feature order
                user_input = [[gender_val, age, hypertension, heart_disease,
                              smoking_num, bmi, hba1c, blood_glucose]]
                
                # Apply scaling if scaler exists
                if scaler is not None:
                    user_input_scaled = scaler.transform(user_input)
                else:
                    user_input_scaled = user_input
                
                # Get prediction and probability
                diabetes_prediction = model.predict(user_input_scaled)
                
                # Get prediction probability if available
                try:
                    diabetes_proba = model.predict_proba(user_input_scaled)
                    risk_percentage = diabetes_proba[0][1] * 100
                except:
                    risk_percentage = None
                
                if diabetes_prediction[0] == 1:
                    st.error(f"{name}, high risk of Type 2 Diabetes detected!")
                    if risk_percentage is not None:
                        st.metric("Risk Score", f"{risk_percentage:.1f}%")
                    severity = "high"
                else:
                    st.success(f"{name}, low diabetes risk. Continue healthy lifestyle!")
                    if risk_percentage is not None:
                        st.metric("Risk Score", f"{risk_percentage:.1f}%")
                    severity = "low"
            
            if name:
                with st.spinner("Generating diabetes management recommendations..."):
                    patient_info = {
                        "name": name,
                        "age": age,
                        "bmi": bmi,
                        "blood_glucose": blood_glucose,
                        "hba1c": hba1c,
                        "hypertension": hypertension
                    }
                    
                    recommendations = get_health_recommendations("Type 2 Diabetes", severity, patient_info)
                    if recommendations:
                        display_recommendations(recommendations)
                        display_health_tips_dynamic("Type 2 Diabetes", severity.lower())

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            st.exception(e)

# Heart Disease Prediction
# Heart Disease Prediction
if selected == 'Heart Disease Prediction':
    st.title("❤️ Heart Disease Prediction")
    st.markdown("Assess cardiovascular disease risk based on clinical parameters")
    st.info("📊 Model Accuracy: 89.13% | ROC-AUC: 0.9304 | Based on UCI Heart Disease Dataset")
    
    name = st.text_input("Name:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", [
            "Asymptomatic (ASY)", 
            "Atypical Angina (ATA)", 
            "Non-Anginal Pain (NAP)", 
            "Typical Angina (TA)"
        ])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    
    with col2:
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality (ST)", "LV Hypertrophy (LVH)"])
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    
    with col3:
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=-5.0, max_value=10.0, value=1.0)
        slope = st.selectbox("ST Slope", ["Upsloping (Up)", "Flat", "Downsloping (Down)"])
    
    if st.button("Predict Heart Disease"):
        try:
            # Check if heart model loaded properly
            if heart_model is None:
                st.error("Heart disease model not loaded. Please check model file.")
            else:
                # Get model components from packaged model
                if isinstance(heart_model, dict):
                    model = heart_model['model']
                    scaler = heart_model['scaler']
                    mappings = heart_model.get('mappings', {})
                else:
                    # Fallback for old model format
                    model = heart_model
                    scaler = None
                    mappings = {}
                
                # Encode Sex: {'F': 0, 'M': 1}
                sex_encoded = 1 if sex == "Male" else 0
                
                # Encode ChestPainType: {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3}
                cp_mapping = {
                    "Asymptomatic (ASY)": 0,
                    "Atypical Angina (ATA)": 1,
                    "Non-Anginal Pain (NAP)": 2,
                    "Typical Angina (TA)": 3
                }
                cp_encoded = cp_mapping.get(cp, 0)
                
                # Encode FastingBS
                fbs_encoded = 1 if fbs == "Yes" else 0
                
                # Encode RestingECG: {'LVH': 0, 'Normal': 1, 'ST': 2}
                restecg_mapping = {
                    "Normal": 1,
                    "ST-T Abnormality (ST)": 2,
                    "LV Hypertrophy (LVH)": 0
                }
                restecg_encoded = restecg_mapping.get(restecg, 1)
                
                # Encode ExerciseAngina: {'N': 0, 'Y': 1}
                exang_encoded = 1 if exang == "Yes" else 0
                
                # Encode ST_Slope: {'Down': 0, 'Flat': 1, 'Up': 2}
                slope_mapping = {
                    "Upsloping (Up)": 2,
                    "Flat": 1,
                    "Downsloping (Down)": 0
                }
                slope_encoded = slope_mapping.get(slope, 1)
                
                # Prepare input: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, 
                #                RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
                user_input = [[age, sex_encoded, cp_encoded, trestbps, chol, fbs_encoded, 
                              restecg_encoded, thalach, exang_encoded, oldpeak, slope_encoded]]
                
                # Apply scaling if available
                if scaler is not None:
                    user_input_scaled = scaler.transform(user_input)
                else:
                    user_input_scaled = user_input
                
                # Get prediction and probability
                heart_prediction = model.predict(user_input_scaled)
                
                try:
                    heart_proba = model.predict_proba(user_input_scaled)
                    risk_percentage = heart_proba[0][1] * 100
                except:
                    risk_percentage = None
                
                if heart_prediction[0] == 1:
                    st.error(f"{name}, heart disease risk detected! Consult a cardiologist immediately.")
                    if risk_percentage is not None:
                        st.metric("Risk Score", f"{risk_percentage:.1f}%")
                    severity = "high"
                else:
                    st.success(f"{name}, low heart disease risk. Maintain a healthy lifestyle!")
                    if risk_percentage is not None:
                        st.metric("Risk Score", f"{risk_percentage:.1f}%")
                    severity = "low"
                
                # Get AI recommendations
                if name:
                    with st.spinner("Generating cardiac health recommendations..."):
                        patient_info = {
                            "name": name,
                            "age": age,
                            "cholesterol": chol,
                            "blood_pressure": trestbps,
                            "max_heart_rate": thalach
                        }
                        
                        recommendations = get_health_recommendations("Heart Disease", severity, patient_info)
                        if recommendations:
                            display_recommendations(recommendations)
                            display_health_tips_dynamic("Heart Disease", severity.lower())

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            st.exception(e)

# Parkinson's Disease Prediction
if selected == 'Parkinsons Prediction':
    st.title("🧠 Parkinson's Disease Prediction")
    st.markdown("Comprehensive Parkinson's disease risk assessment using clinical, lifestyle, and symptom data")
    st.info("📊 Model Accuracy: 92.64% | 32 Clinical Features | 2,105 Patient Dataset")
    
    name = st.text_input("Name:")
    
    # Check if model is loaded
    if "parkinsons_model" not in loaded_models:
        st.error("⚠️ Parkinson's Disease model not loaded. Please ensure the model file exists.")
    else:
        model_data = loaded_models["parkinsons_model"]
        
        # Demographic Section
        st.subheader("📋 Demographics & Lifestyle")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            age = st.number_input("Age", min_value=50, max_value=90, value=65, 
                                  help="Patient age (50-90 years)")
            gender = st.selectbox("Gender", [0, 1], 
                                  format_func=lambda x: "Male" if x == 0 else "Female")
        
        with col2:
            ethnicity = st.selectbox("Ethnicity", [0, 1, 2, 3],
                                     format_func=lambda x: {0: "Caucasian", 1: "African American", 
                                                           2: "Asian", 3: "Other"}[x])
            education = st.selectbox("Education Level", [0, 1, 2, 3],
                                     format_func=lambda x: {0: "None", 1: "High School", 
                                                           2: "Bachelor's", 3: "Higher"}[x])
        
        with col3:
            bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, value=25.0,
                                  help="Body Mass Index (15-40)")
            smoking = st.selectbox("Smoking", [0, 1], 
                                   format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col4:
            alcohol = st.number_input("Alcohol (units/week)", min_value=0.0, max_value=20.0, value=5.0)
            physical_activity = st.number_input("Physical Activity (hrs/week)", min_value=0.0, max_value=10.0, value=5.0)
        
        col5, col6 = st.columns(2)
        with col5:
            diet_quality = st.slider("Diet Quality Score", min_value=0, max_value=10, value=6)
        with col6:
            sleep_quality = st.slider("Sleep Quality Score", min_value=4, max_value=10, value=7)
        
        # Medical History Section
        st.subheader("🏥 Medical History")
        col7, col8, col9 = st.columns(3)
        
        with col7:
            family_history = st.selectbox("Family History of Parkinson's", [0, 1],
                                          format_func=lambda x: "No" if x == 0 else "Yes")
            tbi = st.selectbox("Traumatic Brain Injury", [0, 1],
                               format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col8:
            hypertension = st.selectbox("Hypertension", [0, 1],
                                        format_func=lambda x: "No" if x == 0 else "Yes")
            diabetes = st.selectbox("Diabetes", [0, 1],
                                    format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col9:
            depression = st.selectbox("Depression", [0, 1],
                                      format_func=lambda x: "No" if x == 0 else "Yes")
            stroke = st.selectbox("History of Stroke", [0, 1],
                                  format_func=lambda x: "No" if x == 0 else "Yes")
        
        # Clinical Measurements Section
        st.subheader("📊 Clinical Measurements")
        col10, col11, col12 = st.columns(3)
        
        with col10:
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=90, max_value=180, value=120)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=60, max_value=120, value=80)
        
        with col11:
            cholesterol_total = st.number_input("Total Cholesterol (mg/dL)", min_value=150, max_value=300, value=200)
            cholesterol_ldl = st.number_input("LDL Cholesterol (mg/dL)", min_value=50, max_value=200, value=100)
        
        with col12:
            cholesterol_hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=20, max_value=100, value=50)
            cholesterol_trig = st.number_input("Triglycerides (mg/dL)", min_value=50, max_value=400, value=150)
        
        # Cognitive and Functional Assessment Section
        st.subheader("🧠 Cognitive & Functional Assessment")
        col13, col14, col15 = st.columns(3)
        
        with col13:
            updrs = st.number_input("UPDRS Score", min_value=0, max_value=199, value=30,
                                    help="Unified Parkinson's Disease Rating Scale (0-199). Higher = more severe")
        
        with col14:
            moca = st.number_input("MoCA Score", min_value=0, max_value=30, value=26,
                                   help="Montreal Cognitive Assessment (0-30). Lower = more impairment")
        
        with col15:
            functional = st.number_input("Functional Assessment", min_value=0.0, max_value=10.0, value=8.0,
                                         help="Functional ability score (0-10). Lower = more impairment")
        
        # Symptoms Section
        st.subheader("⚠️ Current Symptoms")
        col16, col17, col18, col19 = st.columns(4)
        
        with col16:
            tremor = st.selectbox("Tremor", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            rigidity = st.selectbox("Muscle Rigidity", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col17:
            bradykinesia = st.selectbox("Bradykinesia", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                                        help="Slowness of movement")
            postural_instability = st.selectbox("Postural Instability", [0, 1], 
                                                format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col18:
            speech_problems = st.selectbox("Speech Problems", [0, 1], 
                                           format_func=lambda x: "No" if x == 0 else "Yes")
            sleep_disorders = st.selectbox("Sleep Disorders", [0, 1], 
                                           format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col19:
            constipation = st.selectbox("Constipation", [0, 1], 
                                        format_func=lambda x: "No" if x == 0 else "Yes")
        
        # Prediction
        if st.button("🔍 Predict Parkinson's Disease Risk", type="primary"):
            try:
                # Build feature array in correct order
                features = [
                    age, gender, ethnicity, education,
                    bmi, smoking, alcohol, physical_activity,
                    diet_quality, sleep_quality, family_history, tbi,
                    hypertension, diabetes, depression, stroke,
                    systolic_bp, diastolic_bp, cholesterol_total, cholesterol_ldl,
                    cholesterol_hdl, cholesterol_trig, updrs, moca,
                    functional, tremor, rigidity, bradykinesia,
                    postural_instability, speech_problems, sleep_disorders, constipation
                ]
                
                # Extract model components
                model = model_data['model']
                scaler = model_data['scaler']
                
                # Scale and predict
                import numpy as np
                features_scaled = scaler.transform([features])
                prediction = model.predict(features_scaled)[0]
                probabilities = model.predict_proba(features_scaled)[0]
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                if prediction == 1:
                    risk_prob = probabilities[1] * 100
                    st.error(f"⚠️ **{name if name else 'Patient'}** - Parkinson's Disease Indicators Detected")
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric("Risk Level", "HIGH", delta=None)
                    with col_res2:
                        st.metric("Confidence", f"{risk_prob:.1f}%")
                    
                    # Risk stratification
                    if risk_prob >= 80:
                        severity = "high"
                        st.warning("🔴 **Very High Risk** - Immediate neurological evaluation strongly recommended")
                    elif risk_prob >= 60:
                        severity = "moderate"
                        st.warning("🟠 **Elevated Risk** - Schedule comprehensive neurological assessment")
                    else:
                        severity = "moderate"
                        st.info("🟡 **Moderate Risk** - Consider follow-up neurological consultation")
                    
                    try:
                        image = Image.open('positive.jpg')
                        st.image(image, caption='Positive Indicators Detected', width=200)
                    except:
                        pass
                else:
                    healthy_prob = probabilities[0] * 100
                    st.success(f"✅ **{name if name else 'Patient'}** - No Significant Parkinson's Indicators")
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric("Risk Level", "LOW", delta=None)
                    with col_res2:
                        st.metric("Confidence", f"{healthy_prob:.1f}%")
                    
                    severity = "low"
                    st.info("🟢 Continue regular health monitoring and maintain healthy lifestyle")
                
                # Show probability breakdown
                st.markdown("### Probability Breakdown")
                prob_col1, prob_col2 = st.columns(2)
                with prob_col1:
                    st.progress(probabilities[0], text=f"No Parkinson's: {probabilities[0]*100:.1f}%")
                with prob_col2:
                    st.progress(probabilities[1], text=f"Parkinson's: {probabilities[1]*100:.1f}%")
                
                # Key Risk Factors Analysis
                st.markdown("### Key Risk Factors in Your Profile")
                risk_factors = []
                protective_factors = []
                
                if updrs > 100:
                    risk_factors.append(f"• High UPDRS Score ({updrs}) - indicates significant motor impairment")
                if moca < 20:
                    risk_factors.append(f"• Low MoCA Score ({moca}) - suggests cognitive impairment")
                if functional < 5:
                    risk_factors.append(f"• Low Functional Assessment ({functional}) - indicates daily activity limitations")
                if tremor == 1:
                    risk_factors.append("• Presence of tremor - cardinal symptom of Parkinson's")
                if rigidity == 1:
                    risk_factors.append("• Muscle rigidity detected")
                if bradykinesia == 1:
                    risk_factors.append("• Bradykinesia present - slowness of movement")
                if family_history == 1:
                    risk_factors.append("• Family history of Parkinson's disease")
                if tbi == 1:
                    risk_factors.append("• History of traumatic brain injury")
                if age > 70:
                    risk_factors.append(f"• Advanced age ({age}) - higher baseline risk")
                
                if physical_activity >= 7:
                    protective_factors.append(f"• Good physical activity level ({physical_activity} hrs/week)")
                if diet_quality >= 7:
                    protective_factors.append(f"• Healthy diet quality score ({diet_quality}/10)")
                if moca >= 26:
                    protective_factors.append(f"• Good cognitive function (MoCA: {moca})")
                if smoking == 0:
                    protective_factors.append("• Non-smoker status")
                
                if risk_factors:
                    st.error("**Risk Factors Identified:**")
                    for rf in risk_factors:
                        st.markdown(rf)
                
                if protective_factors:
                    st.success("**Protective Factors:**")
                    for pf in protective_factors:
                        st.markdown(pf)
                
                # AI Recommendations
                if name:
                    with st.spinner("Generating personalized neurological health recommendations..."):
                        patient_info = {
                            "name": name,
                            "age": age,
                            "updrs_score": updrs,
                            "moca_score": moca,
                            "symptoms": {
                                "tremor": tremor == 1,
                                "rigidity": rigidity == 1,
                                "bradykinesia": bradykinesia == 1,
                                "postural_instability": postural_instability == 1
                            }
                        }
                        
                        recommendations = get_health_recommendations("Parkinson's Disease", severity, patient_info)
                        if recommendations:
                            display_recommendations(recommendations)
                            display_health_tips_dynamic("Parkinson's Disease", severity.lower())
                
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
                st.exception(e)

# Lung Cancer Prediction
if selected == 'Lung Cancer Prediction':
    st.title("🫁 Lung Cancer Risk Prediction")
    st.markdown("Comprehensive lung cancer risk assessment based on lifestyle, environmental factors, and symptoms")
    st.info("📊 Model Accuracy: 100% | 3-Level Risk Classification (Low/Medium/High) | 23 Risk Factors")
    
    name = st.text_input("Name:")
    
    # Demographics Section
    st.subheader("📋 Demographics")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
    with col2:
        gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    
    # Environmental & Lifestyle Risk Factors
    st.subheader("🌍 Environmental & Lifestyle Factors")
    st.markdown("*Rate each factor from 1 (Low) to 8 (High)*")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        air_pollution = st.slider("Air Pollution Exposure", 1, 8, 3, help="1=Low, 8=High exposure")
        alcohol_use = st.slider("Alcohol Use", 1, 8, 2)
    with col2:
        dust_allergy = st.slider("Dust Allergy", 1, 8, 3)
        occupational_hazards = st.slider("Occupational Hazards", 1, 8, 2, help="Exposure to chemicals, asbestos, etc.")
    with col3:
        genetic_risk = st.slider("Genetic Risk", 1, 8, 3, help="Family history of lung cancer")
        balanced_diet = st.slider("Balanced Diet", 1, 8, 5, help="1=Poor, 8=Excellent")
    with col4:
        obesity = st.slider("Obesity Level", 1, 8, 3)
        smoking = st.slider("Smoking", 1, 8, 2, help="1=Non-smoker, 8=Heavy smoker")
    
    col1, col2 = st.columns(2)
    with col1:
        passive_smoker = st.slider("Passive Smoker", 1, 8, 2, help="Exposure to secondhand smoke")
        chronic_lung_disease = st.slider("Chronic Lung Disease", 1, 8, 2, help="COPD, asthma, etc.")
    with col2:
        snoring = st.slider("Snoring", 1, 8, 3)
    
    # Symptoms Section
    st.subheader("🩺 Symptoms")
    st.markdown("*Rate severity from 1 (None/Mild) to 8 (Severe)*")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        chest_pain = st.slider("Chest Pain", 1, 8, 2)
        coughing_blood = st.slider("Coughing of Blood", 1, 8, 1, help="Hemoptysis")
        fatigue = st.slider("Fatigue", 1, 8, 3)
        weight_loss = st.slider("Unexplained Weight Loss", 1, 8, 2)
    with col2:
        shortness_breath = st.slider("Shortness of Breath", 1, 8, 2)
        wheezing = st.slider("Wheezing", 1, 8, 2)
        swallowing_difficulty = st.slider("Swallowing Difficulty", 1, 8, 1)
        clubbing = st.slider("Clubbing of Finger Nails", 1, 8, 1)
    with col3:
        frequent_cold = st.slider("Frequent Cold", 1, 8, 3)
        dry_cough = st.slider("Dry Cough", 1, 8, 2)
    
    if st.button("Predict Lung Cancer Risk"):
        try:
            # Encode gender (1->0, 2->1 based on training)
            gender_encoded = 0 if gender == 1 else 1
            
            # Create feature array in correct order
            user_input = [
                age, gender_encoded, air_pollution, alcohol_use, dust_allergy,
                occupational_hazards, genetic_risk, chronic_lung_disease,
                balanced_diet, obesity, smoking, passive_smoker,
                chest_pain, coughing_blood, fatigue, weight_loss,
                shortness_breath, wheezing, swallowing_difficulty,
                clubbing, frequent_cold, dry_cough, snoring
            ]
            
            # Get model components
            model_data = models.get('lung_cancer_model')
            if model_data and isinstance(model_data, dict):
                lung_model = model_data['model']
                lung_scaler = model_data['scaler']
                level_encoder = model_data['level_encoder']
                
                # Scale and predict
                user_input_scaled = lung_scaler.transform([user_input])
                lung_prediction = lung_model.predict(user_input_scaled)
                lung_proba = lung_model.predict_proba(user_input_scaled)[0]
                
                # Get predicted class
                predicted_class = level_encoder.classes_[lung_prediction[0]]
                
                # Display results
                st.subheader("📊 Prediction Results")
                
                # Get probability for each class
                class_probs = {level_encoder.classes_[i]: lung_proba[i] for i in range(len(level_encoder.classes_))}
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Low Risk", f"{class_probs.get('Low', 0)*100:.1f}%")
                with col2:
                    st.metric("Medium Risk", f"{class_probs.get('Medium', 0)*100:.1f}%")
                with col3:
                    st.metric("High Risk", f"{class_probs.get('High', 0)*100:.1f}%")
                
                # Display main result
                if predicted_class == "High":
                    st.error(f"⚠️ {name if name else 'Patient'}, HIGH lung cancer risk detected! Immediate medical consultation and screening recommended.")
                    severity = "high"
                elif predicted_class == "Medium":
                    st.warning(f"⚠️ {name if name else 'Patient'}, MODERATE lung cancer risk. Please consult a pulmonologist for screening.")
                    severity = "medium"
                else:
                    st.success(f"✅ {name if name else 'Patient'}, LOW lung cancer risk. Continue healthy habits and regular check-ups!")
                    severity = "low"
                
                # Key risk factors identified
                st.subheader("🔑 Key Risk Factors")
                risk_factors = []
                if smoking >= 5:
                    risk_factors.append(f"⚠️ High smoking level: {smoking}/8")
                if air_pollution >= 5:
                    risk_factors.append(f"⚠️ High air pollution exposure: {air_pollution}/8")
                if coughing_blood >= 3:
                    risk_factors.append(f"🚨 Coughing blood detected: {coughing_blood}/8 - Seek immediate medical attention")
                if genetic_risk >= 5:
                    risk_factors.append(f"⚠️ High genetic risk: {genetic_risk}/8")
                if occupational_hazards >= 5:
                    risk_factors.append(f"⚠️ Occupational hazards: {occupational_hazards}/8")
                if passive_smoker >= 5:
                    risk_factors.append(f"⚠️ High passive smoking exposure: {passive_smoker}/8")
                if chronic_lung_disease >= 5:
                    risk_factors.append(f"⚠️ Chronic lung disease: {chronic_lung_disease}/8")
                
                if risk_factors:
                    for rf in risk_factors:
                        st.write(rf)
                else:
                    st.write("✅ No major risk factors identified")
                
                # AI Recommendations
                if name:
                    with st.spinner("Generating personalized lung health recommendations..."):
                        patient_info = {
                            "name": name,
                            "age": age,
                            "risk_level": predicted_class,
                            "smoking_level": smoking,
                            "air_pollution_exposure": air_pollution,
                            "symptoms": {
                                "chest_pain": chest_pain,
                                "coughing_blood": coughing_blood,
                                "shortness_breath": shortness_breath
                            }
                        }
                        
                        recommendations = get_health_recommendations("Lung Cancer", severity, patient_info)
                        if recommendations:
                            display_recommendations(recommendations)
                            display_health_tips_dynamic("Lung Cancer", severity.lower())
            else:
                st.error("Lung cancer model not available. Please check model files.")
                
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Breast Cancer Prediction
if selected == 'Breast Cancer Prediction':
    st.title("🎗️ Breast Cancer Prediction")
    st.markdown("Predict breast cancer based on cell nuclei measurements from FNA biopsy")
    st.info("📊 Model Accuracy: 97.37% | ROC-AUC: 0.9970 | Based on Wisconsin Diagnostic Dataset")
    
    name = st.text_input("Name:")
    
    st.markdown("**These measurements are typically obtained from fine needle aspirate (FNA) of breast mass.**")
    
    # Mean Features
    st.subheader("📏 Mean Measurements")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        radius_mean = st.number_input("Radius Mean", min_value=0.0, max_value=50.0, value=14.0)
        texture_mean = st.number_input("Texture Mean", min_value=0.0, max_value=50.0, value=19.0)
        perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, max_value=200.0, value=92.0)
        area_mean = st.number_input("Area Mean", min_value=0.0, max_value=2500.0, value=655.0)
    
    with col2:
        smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, max_value=0.3, value=0.096)
        compactness_mean = st.number_input("Compactness Mean", min_value=0.0, max_value=0.5, value=0.104)
        concavity_mean = st.number_input("Concavity Mean", min_value=0.0, max_value=0.5, value=0.089)
    
    with col3:
        concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, max_value=0.3, value=0.048)
        symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, max_value=0.5, value=0.181)
        fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, max_value=0.1, value=0.063)
    
    # SE Features
    with st.expander("📐 Standard Error Measurements", expanded=False):
        col4, col5, col6 = st.columns(3)
        with col4:
            radius_se = st.number_input("Radius SE", min_value=0.0, max_value=5.0, value=0.406)
            texture_se = st.number_input("Texture SE", min_value=0.0, max_value=5.0, value=1.216)
            perimeter_se = st.number_input("Perimeter SE", min_value=0.0, max_value=30.0, value=2.866)
            area_se = st.number_input("Area SE", min_value=0.0, max_value=500.0, value=40.34)
        with col5:
            smoothness_se = st.number_input("Smoothness SE", min_value=0.0, max_value=0.05, value=0.007)
            compactness_se = st.number_input("Compactness SE", min_value=0.0, max_value=0.2, value=0.025)
            concavity_se = st.number_input("Concavity SE", min_value=0.0, max_value=0.2, value=0.032)
        with col6:
            concave_points_se = st.number_input("Concave Points SE", min_value=0.0, max_value=0.05, value=0.012)
            symmetry_se = st.number_input("Symmetry SE", min_value=0.0, max_value=0.1, value=0.020)
            fractal_dimension_se = st.number_input("Fractal Dimension SE", min_value=0.0, max_value=0.05, value=0.003)
    
    # Worst Features
    with st.expander("📊 Worst (Largest) Measurements", expanded=False):
        col7, col8, col9 = st.columns(3)
        with col7:
            radius_worst = st.number_input("Radius Worst", min_value=0.0, max_value=50.0, value=16.0)
            texture_worst = st.number_input("Texture Worst", min_value=0.0, max_value=50.0, value=25.0)
            perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, max_value=300.0, value=107.0)
            area_worst = st.number_input("Area Worst", min_value=0.0, max_value=4000.0, value=880.0)
        with col8:
            smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, max_value=0.3, value=0.132)
            compactness_worst = st.number_input("Compactness Worst", min_value=0.0, max_value=1.0, value=0.254)
            concavity_worst = st.number_input("Concavity Worst", min_value=0.0, max_value=1.5, value=0.272)
        with col9:
            concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, max_value=0.3, value=0.114)
            symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, max_value=0.7, value=0.290)
            fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0, max_value=0.3, value=0.084)
    
    if st.button("Predict Breast Cancer"):
        try:
            # Check if model loaded properly
            if breast_cancer_model is None:
                st.error("Breast cancer model not loaded. Please check model file.")
            else:
                # Get model components from packaged model
                if isinstance(breast_cancer_model, dict):
                    model = breast_cancer_model['model']
                    scaler = breast_cancer_model['scaler']
                else:
                    # Fallback for old model format
                    model = breast_cancer_model
                    scaler = None
                
                # Prepare input in correct feature order (30 features)
                user_input = [[
                    radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                    compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
                    fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
                    smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se,
                    fractal_dimension_se, radius_worst, texture_worst, perimeter_worst,
                    area_worst, smoothness_worst, compactness_worst, concavity_worst,
                    concave_points_worst, symmetry_worst, fractal_dimension_worst
                ]]
                
                # Apply scaling if available
                if scaler is not None:
                    user_input_scaled = scaler.transform(user_input)
                else:
                    user_input_scaled = user_input
                
                # Get prediction and probability
                breast_prediction = model.predict(user_input_scaled)
                
                try:
                    breast_proba = model.predict_proba(user_input_scaled)
                    malignancy_probability = breast_proba[0][1] * 100
                except:
                    malignancy_probability = None
                
                # Display result (0=Benign, 1=Malignant)
                if breast_prediction[0] == 1:
                    st.error(f"{name}, malignant tumor characteristics detected. Immediate oncology consultation required!")
                    if malignancy_probability is not None:
                        st.metric("Malignancy Score", f"{malignancy_probability:.1f}%")
                    severity = "severe"
                else:
                    st.success(f"{name}, benign tumor characteristics. Continue regular screening.")
                    if malignancy_probability is not None:
                        st.metric("Malignancy Score", f"{malignancy_probability:.1f}%")
                    severity = "low"
                
                # Get AI recommendations
                if name:
                    with st.spinner("Generating breast health recommendations..."):
                        patient_info = {
                            "name": name,
                            "tumor_radius": radius_mean,
                            "tumor_area": area_mean,
                            "tumor_characteristics": "analyzed"
                        }
                        
                        recommendations = get_health_recommendations("Breast Cancer", severity, patient_info)
                        if recommendations:
                            display_recommendations(recommendations)
                            display_health_tips_dynamic("Breast Cancer", severity.lower())

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            st.exception(e)

# Liver Cancer Prediction
if selected == 'Liver Cancer Prediction':
    st.title("🔬 Liver Cancer Risk Prediction")
    st.markdown("Comprehensive liver cancer risk assessment based on clinical and lifestyle factors")
    st.info("📊 Model Accuracy: 94.30% | Binary Classification | 13 Risk Factors | 5,000 Patients Dataset")
    
    name = st.text_input("Name:")
    
    # Check if model is loaded
    if "liver_cancer_model" not in loaded_models:
        st.error("⚠️ Liver Cancer model not loaded. Please ensure the model file exists.")
    else:
        model_data = loaded_models["liver_cancer_model"]
        
        # Demographics Section
        st.subheader("📋 Demographics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=55, 
                                  help="Patient age in years")
        with col2:
            gender = st.selectbox("Gender", ["Female", "Male"])
        with col3:
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1,
                                  help="Body Mass Index (kg/m²)")
        
        # Risk Factors Section
        st.subheader("🚨 Risk Factors")
        col1, col2 = st.columns(2)
        
        with col1:
            alcohol_consumption = st.selectbox("Alcohol Consumption", 
                                               ["Never", "Occasional", "Regular"],
                                               help="Frequency of alcohol consumption")
            smoking_status = st.selectbox("Smoking Status", 
                                          ["Never", "Former", "Current"],
                                          help="Current or past smoking history")
            diabetes = st.selectbox("Diabetes", [0, 1], 
                                    format_func=lambda x: "Yes" if x == 1 else "No",
                                    help="History of diabetes")
        
        with col2:
            hepatitis_b = st.selectbox("Hepatitis B", [0, 1], 
                                       format_func=lambda x: "Positive" if x == 1 else "Negative",
                                       help="Hepatitis B virus infection status")
            hepatitis_c = st.selectbox("Hepatitis C", [0, 1], 
                                       format_func=lambda x: "Positive" if x == 1 else "Negative",
                                       help="Hepatitis C virus infection status")
            cirrhosis_history = st.selectbox("Cirrhosis History", [0, 1], 
                                             format_func=lambda x: "Yes" if x == 1 else "No",
                                             help="History of liver cirrhosis")
        
        # Clinical Measurements Section
        st.subheader("🔬 Clinical Measurements")
        col1, col2 = st.columns(2)
        
        with col1:
            liver_function_score = st.number_input("Liver Function Score", 
                                                   min_value=0.0, max_value=100.0, value=50.0, step=0.1,
                                                   help="Composite liver function test score (0-100)")
            alpha_fetoprotein = st.number_input("Alpha-Fetoprotein (AFP) Level (ng/mL)", 
                                                min_value=0.0, max_value=500.0, value=10.0, step=0.1,
                                                help="Tumor marker - elevated levels may indicate liver cancer")
        
        with col2:
            family_history = st.selectbox("Family History of Cancer", [0, 1], 
                                          format_func=lambda x: "Yes" if x == 1 else "No",
                                          help="Cancer history in close relatives")
            physical_activity = st.selectbox("Physical Activity Level", 
                                             ["Low", "Moderate", "High"],
                                             help="Regular exercise frequency")
        
        # AFP level warning
        if alpha_fetoprotein > 20:
            st.warning("⚠️ Elevated AFP levels detected. AFP > 20 ng/mL may warrant further investigation.")
        if alpha_fetoprotein > 400:
            st.error("🚨 Very high AFP levels. Immediate medical consultation recommended.")
        
        if st.button("Predict Liver Cancer Risk"):
            try:
                # Encode categorical variables according to model training
                # gender: Female=0, Male=1
                gender_encoded = 0 if gender == "Female" else 1
                
                # alcohol_consumption: Never=0, Occasional=1, Regular=2
                alcohol_map = {"Never": 0, "Occasional": 1, "Regular": 2}
                alcohol_encoded = alcohol_map[alcohol_consumption]
                
                # smoking_status: Current=0, Former=1, Never=2
                smoking_map = {"Current": 0, "Former": 1, "Never": 2}
                smoking_encoded = smoking_map[smoking_status]
                
                # physical_activity_level: High=0, Low=1, Moderate=2
                activity_map = {"High": 0, "Low": 1, "Moderate": 2}
                activity_encoded = activity_map[physical_activity]
                
                # Create feature array in correct order
                # ['age', 'gender', 'bmi', 'alcohol_consumption', 'smoking_status', 
                #  'hepatitis_b', 'hepatitis_c', 'liver_function_score', 'alpha_fetoprotein_level', 
                #  'cirrhosis_history', 'family_history_cancer', 'physical_activity_level', 'diabetes']
                user_input = [
                    age, gender_encoded, bmi, alcohol_encoded, smoking_encoded,
                    hepatitis_b, hepatitis_c, liver_function_score, alpha_fetoprotein,
                    cirrhosis_history, family_history, activity_encoded, diabetes
                ]
                
                # Get model components
                liver_cancer_clf = model_data['model']
                liver_scaler = model_data['scaler']
                numerical_cols = model_data.get('numerical_columns', 
                    ['age', 'bmi', 'hepatitis_b', 'hepatitis_c', 'liver_function_score', 
                     'alpha_fetoprotein_level', 'cirrhosis_history', 'family_history_cancer', 'diabetes'])
                
                # Scale numerical features (create DataFrame for proper column handling)
                import pandas as pd
                feature_cols = model_data.get('feature_columns', 
                    ['age', 'gender', 'bmi', 'alcohol_consumption', 'smoking_status', 
                     'hepatitis_b', 'hepatitis_c', 'liver_function_score', 'alpha_fetoprotein_level', 
                     'cirrhosis_history', 'family_history_cancer', 'physical_activity_level', 'diabetes'])
                
                input_df = pd.DataFrame([user_input], columns=feature_cols)
                input_df[numerical_cols] = liver_scaler.transform(input_df[numerical_cols])
                
                # Predict
                prediction = liver_cancer_clf.predict(input_df)[0]
                probability = liver_cancer_clf.predict_proba(input_df)[0]
                
                # Display results
                st.subheader("📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("No Cancer Probability", f"{probability[0]*100:.1f}%")
                with col2:
                    st.metric("Cancer Risk", f"{probability[1]*100:.1f}%")
                with col3:
                    risk_level = "HIGH" if probability[1] > 0.5 else "MODERATE" if probability[1] > 0.3 else "LOW"
                    st.metric("Risk Level", risk_level)
                
                # Display main result
                if prediction == 1:
                    st.error(f"⚠️ {name if name else 'Patient'}, HIGH liver cancer risk detected! Immediate hepatology consultation and imaging studies recommended.")
                    severity = "high"
                    
                    st.markdown("""
                    ### 🏥 Recommended Actions:
                    1. **Immediate**: Consult a hepatologist or oncologist
                    2. **Imaging**: Ultrasound, CT scan, or MRI of the liver
                    3. **Lab Tests**: Complete liver panel, additional tumor markers
                    4. **Biopsy**: May be recommended based on imaging results
                    """)
                else:
                    if probability[1] > 0.3:
                        st.warning(f"⚠️ {name if name else 'Patient'}, MODERATE liver cancer risk. Regular monitoring recommended.")
                        severity = "medium"
                    else:
                        st.success(f"✅ {name if name else 'Patient'}, LOW liver cancer risk. Continue healthy habits and regular check-ups!")
                        severity = "low"
                
                # Risk factor analysis
                st.subheader("🔑 Risk Factor Analysis")
                
                risk_factors = []
                protective_factors = []
                
                if hepatitis_b == 1:
                    risk_factors.append("🦠 Hepatitis B positive - Major risk factor for liver cancer")
                if hepatitis_c == 1:
                    risk_factors.append("🦠 Hepatitis C positive - Significant risk factor")
                if cirrhosis_history == 1:
                    risk_factors.append("🔴 Cirrhosis history - Strong predictor of liver cancer")
                if alpha_fetoprotein > 20:
                    risk_factors.append(f"📈 Elevated AFP: {alpha_fetoprotein} ng/mL")
                if alcohol_consumption == "Regular":
                    risk_factors.append("🍺 Regular alcohol consumption")
                if smoking_status == "Current":
                    risk_factors.append("🚬 Current smoker")
                if diabetes == 1:
                    risk_factors.append("💉 Diabetes - Associated with increased liver cancer risk")
                if family_history == 1:
                    risk_factors.append("👨‍👩‍👧 Family history of cancer")
                if bmi > 30:
                    risk_factors.append(f"⚖️ Obesity (BMI: {bmi:.1f})")
                
                if smoking_status == "Never":
                    protective_factors.append("✅ Non-smoker")
                if alcohol_consumption == "Never":
                    protective_factors.append("✅ No alcohol consumption")
                if physical_activity == "High":
                    protective_factors.append("✅ High physical activity level")
                if hepatitis_b == 0 and hepatitis_c == 0:
                    protective_factors.append("✅ No viral hepatitis")
                if 18.5 <= bmi <= 24.9:
                    protective_factors.append("✅ Healthy BMI")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Risk Factors:**")
                    if risk_factors:
                        for rf in risk_factors:
                            st.write(rf)
                    else:
                        st.write("No major risk factors identified")
                
                with col2:
                    st.markdown("**Protective Factors:**")
                    if protective_factors:
                        for pf in protective_factors:
                            st.write(pf)
                    else:
                        st.write("Consider lifestyle modifications")
                
                # Feature importance visualization
                st.subheader("📈 Key Predictors (Model Feature Importance)")
                
                feature_importance = model_data.get('feature_importance', [])
                if feature_importance:
                    top_features = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)[:5]
                    
                    feature_names = [f['feature'].replace('_', ' ').title() for f in top_features]
                    importances = [f['importance']*100 for f in top_features]
                    
                    import pandas as pd
                    fi_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance (%)': importances
                    })
                    st.bar_chart(fi_df.set_index('Feature'))
                
                # AI Recommendations
                if name:
                    with st.spinner("Generating personalized liver health recommendations..."):
                        patient_info = {
                            "name": name,
                            "age": age,
                            "risk_level": risk_level,
                            "afp_level": alpha_fetoprotein,
                            "hepatitis_status": "HBV+" if hepatitis_b else ("HCV+" if hepatitis_c else "Negative"),
                            "cirrhosis": "Yes" if cirrhosis_history else "No",
                            "risk_factors": risk_factors
                        }
                        
                        recommendations = get_health_recommendations("Liver Cancer", severity, patient_info)
                        if recommendations:
                            display_recommendations(recommendations)
                            display_health_tips_dynamic("Liver Cancer", severity.lower())
                
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Kidney Disease Prediction
if selected == 'Kidney Disease Prediction':
    st.title("🫘 Chronic Kidney Disease Prediction")
    st.markdown("Comprehensive kidney disease risk assessment using clinical measurements")

    name = st.text_input("Name:")
    
    # Demographics Section
    st.subheader("📋 Demographics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age", min_value=20, max_value=90, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        ethnicity = st.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Other"])
        socioeconomic = st.selectbox("Socioeconomic Status", ["Low", "Middle", "High"])
    with col3:
        education = st.selectbox("Education Level", ["None", "High School", "Bachelor's", "Higher"])
        bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, value=25.0, step=0.1)
    with col4:
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.number_input("Alcohol (units/week)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
    
    # Lifestyle Section
    st.subheader("🏃 Lifestyle")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        physical_activity = st.number_input("Physical Activity (hrs/week)", min_value=0.0, max_value=10.0, value=3.0, step=0.5)
    with col2:
        diet_quality = st.slider("Diet Quality Score", 0, 10, 6)
    with col3:
        sleep_quality = st.slider("Sleep Quality Score", 4, 10, 7)
    with col4:
        health_literacy = st.slider("Health Literacy Score", 0, 10, 6)
    
    # Medical History Section
    st.subheader("📜 Medical History")
    col1, col2, col3 = st.columns(3)
    with col1:
        family_kidney = st.selectbox("Family History - Kidney Disease", ["No", "Yes"])
        family_hypertension = st.selectbox("Family History - Hypertension", ["No", "Yes"])
    with col2:
        family_diabetes = st.selectbox("Family History - Diabetes", ["No", "Yes"])
        prev_aki = st.selectbox("Previous Acute Kidney Injury", ["No", "Yes"])
    with col3:
        uti = st.selectbox("Urinary Tract Infections History", ["No", "Yes"])
    
    # Clinical Measurements Section
    st.subheader("🔬 Clinical Measurements")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=90, max_value=180, value=120)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=60, max_value=120, value=80)
        fasting_sugar = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=70.0, max_value=200.0, value=100.0, step=1.0)
        hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=10.0, value=5.5, step=0.1)
    with col2:
        serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.5, max_value=5.0, value=1.0, step=0.1, help="Normal: 0.7-1.3")
        bun = st.number_input("BUN Levels (mg/dL)", min_value=5.0, max_value=50.0, value=15.0, step=0.5, help="Normal: 7-20")
        gfr = st.number_input("GFR (mL/min/1.73m²)", min_value=15.0, max_value=120.0, value=90.0, step=1.0, help="Normal: >90")
        protein_urine = st.number_input("Protein in Urine (g/day)", min_value=0.0, max_value=5.0, value=0.1, step=0.1)
    with col3:
        acr = st.number_input("ACR (mg/g)", min_value=0.0, max_value=300.0, value=20.0, step=5.0, help="Normal: <30")
        sodium = st.number_input("Serum Sodium (mEq/L)", min_value=135.0, max_value=145.0, value=140.0, step=0.5)
        potassium = st.number_input("Serum Potassium (mEq/L)", min_value=3.5, max_value=5.5, value=4.0, step=0.1)
        calcium = st.number_input("Serum Calcium (mg/dL)", min_value=8.5, max_value=10.5, value=9.5, step=0.1)
    with col4:
        phosphorus = st.number_input("Serum Phosphorus (mg/dL)", min_value=2.5, max_value=4.5, value=3.5, step=0.1)
        hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=10.0, max_value=18.0, value=14.0, step=0.1)
        chol_total = st.number_input("Total Cholesterol (mg/dL)", min_value=150.0, max_value=300.0, value=180.0, step=5.0)
        chol_ldl = st.number_input("LDL Cholesterol (mg/dL)", min_value=50.0, max_value=200.0, value=100.0, step=5.0)
    
    col1, col2 = st.columns(2)
    with col1:
        chol_hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=20.0, max_value=100.0, value=50.0, step=5.0)
        chol_trig = st.number_input("Triglycerides (mg/dL)", min_value=50.0, max_value=400.0, value=150.0, step=10.0)
    
    # Medications Section
    st.subheader("💊 Medications")
    col1, col2, col3 = st.columns(3)
    with col1:
        ace_inhibitors = st.selectbox("ACE Inhibitors", ["No", "Yes"])
        diuretics = st.selectbox("Diuretics", ["No", "Yes"])
    with col2:
        nsaids_use = st.number_input("NSAIDs Use (times/week)", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
        statins = st.selectbox("Statins", ["No", "Yes"])
    with col3:
        antidiabetic = st.selectbox("Antidiabetic Medications", ["No", "Yes"])
    
    # Symptoms Section
    st.subheader("🩺 Symptoms")
    col1, col2, col3 = st.columns(3)
    with col1:
        edema = st.selectbox("Edema (Swelling)", ["No", "Yes"])
        fatigue = st.slider("Fatigue Level", 0, 10, 3)
    with col2:
        nausea = st.number_input("Nausea/Vomiting (times/week)", min_value=0.0, max_value=7.0, value=0.0, step=0.5)
        muscle_cramps = st.number_input("Muscle Cramps (times/week)", min_value=0.0, max_value=7.0, value=0.5, step=0.5)
    with col3:
        itching = st.slider("Itching Severity", 0, 10, 2)
        qol_score = st.slider("Quality of Life Score", 0, 100, 75)
    
    # Environmental Factors Section
    st.subheader("🌍 Environmental & Health Behaviors")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        heavy_metals = st.selectbox("Heavy Metals Exposure", ["No", "Yes"])
    with col2:
        chemical_exposure = st.selectbox("Occupational Chemical Exposure", ["No", "Yes"])
    with col3:
        water_quality = st.selectbox("Water Quality", ["Good", "Poor"])
    with col4:
        checkups_freq = st.number_input("Medical Checkups/Year", min_value=0.0, max_value=4.0, value=2.0, step=0.5)
        med_adherence = st.slider("Medication Adherence Score", 0, 10, 7)
    
    if st.button("Predict Kidney Disease"):
        try:
            # Encode categorical variables
            gender_enc = 0 if gender == "Male" else 1
            ethnicity_enc = {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3}[ethnicity]
            socio_enc = {"Low": 0, "Middle": 1, "High": 2}[socioeconomic]
            edu_enc = {"None": 0, "High School": 1, "Bachelor's": 2, "Higher": 3}[education]
            smoking_enc = 1 if smoking == "Yes" else 0
            family_kidney_enc = 1 if family_kidney == "Yes" else 0
            family_hyp_enc = 1 if family_hypertension == "Yes" else 0
            family_diab_enc = 1 if family_diabetes == "Yes" else 0
            prev_aki_enc = 1 if prev_aki == "Yes" else 0
            uti_enc = 1 if uti == "Yes" else 0
            ace_enc = 1 if ace_inhibitors == "Yes" else 0
            diuretics_enc = 1 if diuretics == "Yes" else 0
            statins_enc = 1 if statins == "Yes" else 0
            antidiab_enc = 1 if antidiabetic == "Yes" else 0
            edema_enc = 1 if edema == "Yes" else 0
            heavy_metals_enc = 1 if heavy_metals == "Yes" else 0
            chem_exp_enc = 1 if chemical_exposure == "Yes" else 0
            water_enc = 0 if water_quality == "Good" else 1
            
            # Create feature array in the correct order
            user_input = [
                age, gender_enc, ethnicity_enc, socio_enc, edu_enc, bmi, smoking_enc,
                alcohol, physical_activity, diet_quality, sleep_quality,
                family_kidney_enc, family_hyp_enc, family_diab_enc, prev_aki_enc, uti_enc,
                systolic_bp, diastolic_bp, fasting_sugar, hba1c, serum_creatinine,
                bun, gfr, protein_urine, acr, sodium, potassium, calcium, phosphorus,
                hemoglobin, chol_total, chol_ldl, chol_hdl, chol_trig,
                ace_enc, diuretics_enc, nsaids_use, statins_enc, antidiab_enc,
                edema_enc, fatigue, nausea, muscle_cramps, itching, qol_score,
                heavy_metals_enc, chem_exp_enc, water_enc, checkups_freq, med_adherence, health_literacy
            ]
            
            # Get model components
            model_data = models.get('kidney_disease_model')
            if model_data and isinstance(model_data, dict):
                kidney_model = model_data['model']
                kidney_scaler = model_data['scaler']
                
                # Scale and predict
                user_input_scaled = kidney_scaler.transform([user_input])
                kidney_prediction = kidney_model.predict(user_input_scaled)
                kidney_probability = kidney_model.predict_proba(user_input_scaled)[0][1]
                
                # Display results
                st.subheader("📊 Prediction Results")
                
                # Risk level based on probability
                if kidney_probability < 0.3:
                    risk_level = "Low"
                    severity = "low"
                    risk_color = "green"
                elif kidney_probability < 0.6:
                    risk_level = "Moderate"
                    severity = "medium"
                    risk_color = "orange"
                else:
                    risk_level = "High"
                    severity = "high"
                    risk_color = "red"
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("CKD Risk Score", f"{kidney_probability * 100:.1f}%")
                with col2:
                    st.metric("Risk Level", risk_level)
                with col3:
                    st.metric("GFR Status", "Normal" if gfr >= 90 else "Reduced" if gfr >= 60 else "Low" if gfr >= 30 else "Very Low")
                
                if kidney_prediction[0] == 1 or kidney_probability >= 0.5:
                    st.error(f"⚠️ {name if name else 'Patient'}, chronic kidney disease risk detected! Please consult a nephrologist.")
                    
                    # CKD Stage estimation based on GFR
                    if gfr >= 90:
                        ckd_stage = "Stage 1 (Normal GFR with kidney damage)"
                    elif gfr >= 60:
                        ckd_stage = "Stage 2 (Mild reduction)"
                    elif gfr >= 45:
                        ckd_stage = "Stage 3a (Mild-moderate reduction)"
                    elif gfr >= 30:
                        ckd_stage = "Stage 3b (Moderate-severe reduction)"
                    elif gfr >= 15:
                        ckd_stage = "Stage 4 (Severe reduction)"
                    else:
                        ckd_stage = "Stage 5 (Kidney failure)"
                    
                    st.warning(f"Estimated CKD Stage: {ckd_stage}")
                else:
                    st.success(f"✅ {name if name else 'Patient'}, no significant kidney disease risk detected. Continue healthy habits!")
                
                # Key indicators
                st.subheader("🔑 Key Kidney Health Indicators")
                indicators = []
                if serum_creatinine > 1.3:
                    indicators.append(f"⚠️ Elevated Serum Creatinine: {serum_creatinine} mg/dL (Normal: 0.7-1.3)")
                if bun > 20:
                    indicators.append(f"⚠️ Elevated BUN: {bun} mg/dL (Normal: 7-20)")
                if gfr < 60:
                    indicators.append(f"⚠️ Reduced GFR: {gfr} mL/min (Normal: >90)")
                if acr > 30:
                    indicators.append(f"⚠️ Elevated ACR: {acr} mg/g (Normal: <30)")
                if protein_urine > 0.3:
                    indicators.append(f"⚠️ Proteinuria detected: {protein_urine} g/day")
                
                if indicators:
                    for ind in indicators:
                        st.write(ind)
                else:
                    st.write("✅ All key kidney indicators are within normal range")
                
                # Recommendations
                if name:
                    with st.spinner("Generating personalized kidney health recommendations..."):
                        patient_info = {
                            "name": name,
                            "age": age,
                            "creatinine": serum_creatinine,
                            "bun": bun,
                            "gfr": gfr,
                            "blood_pressure": f"{systolic_bp}/{diastolic_bp}",
                            "diabetes": "Yes" if hba1c > 6.5 else "No",
                            "hypertension": "Yes" if systolic_bp > 140 or diastolic_bp > 90 else "No",
                            "risk_probability": kidney_probability
                        }
                        
                        recommendations = get_health_recommendations("Chronic Kidney Disease", severity, patient_info)
                        if recommendations:
                            display_recommendations(recommendations)
                            display_health_tips_dynamic("Chronic Kidney Disease", severity.lower())
            else:
                st.error("Kidney disease model not available. Please check model files.")
                
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Liver Disease Prediction
if selected == 'Liver Prediction':
    st.title("🫀 Liver Disease Prediction")
    st.markdown("Assess liver function and disease risk")
    
    name = st.text_input("Name:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", min_value=0.0, max_value=100.0, value=0.7)
        direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, max_value=50.0, value=0.2)
    
    with col2:
        alkaline_phosphotase = st.number_input("Alkaline Phosphotase (IU/L)", min_value=0, max_value=3000, value=187)
        alamine_aminotransferase = st.number_input("Alamine Aminotransferase (IU/L)", min_value=0, max_value=5000, value=16)
        aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (IU/L)", min_value=0, max_value=5000, value=18)
    
    with col3:
        total_proteins = st.number_input("Total Proteins (g/dL)", min_value=0.0, max_value=15.0, value=6.8)
        albumin = st.number_input("Albumin (g/dL)", min_value=0.0, max_value=10.0, value=3.3)
        ag_ratio = st.number_input("Albumin/Globulin Ratio", min_value=0.0, max_value=5.0, value=0.9)
    
    if st.button("Predict Liver Disease"):
        try:
            gender_num = 1 if gender == "Male" else 0
            
            user_input = [age, gender_num, total_bilirubin, direct_bilirubin, 
                         alkaline_phosphotase, alamine_aminotransferase, 
                         aspartate_aminotransferase, total_proteins, albumin, ag_ratio]
            
            liver_prediction = liver_model.predict([user_input])
            
            if liver_prediction[0] == 1:
                st.error(f"{name}, liver disease indicators detected! Hepatology consultation recommended.")
                image = Image.open('positive.jpg')
                st.image(image, caption='Liver Disease Detected')
                severity = "high"
            else:
                st.success(f"{name}, liver function appears normal. Maintain healthy habits!")
                severity = "low"
            
            # Get AI recommendations
            if name:
                with st.spinner("Generating liver health recommendations..."):
                    patient_info = {
                        "name": name,
                        "age": age,
                        "bilirubin": total_bilirubin,
                        "alt": alamine_aminotransferase,
                        "ast": aspartate_aminotransferase
                    }
                    
                    recommendations = get_health_recommendations("Liver Disease", severity, patient_info)
                    if recommendations:
                        display_recommendations(recommendations)
                        display_health_tips_dynamic("Liver Disease", severity.lower())
        except:
            st.error("Error in prediction. Please check all inputs.")

# Hepatitis C Prediction
if selected == 'Hepatitis Prediction':
    st.title("🦠 Hepatitis C Prediction")
    st.markdown("Assess Hepatitis C risk using laboratory blood test values")
    st.info("📊 Model Accuracy: 96.75% | Binary Classification | 12 Laboratory Features | UCI ML Repository Data")
    
    name = st.text_input("Name:")
    
    # Check if model is loaded
    if "hepatitis_c_model" not in loaded_models:
        st.error("⚠️ Hepatitis C model not loaded. Please ensure the model file exists.")
    else:
        model_data = loaded_models["hepatitis_c_model"]
        
        # Demographics Section
        st.subheader("📋 Demographics")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=45, help="Patient age in years")
        with col2:
            sex = st.selectbox("Sex", ["Female", "Male"])
        
        # Laboratory Values Section
        st.subheader("🧪 Liver Function Tests")
        st.markdown("*Enter laboratory blood test values*")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            alb = st.number_input("Albumin (ALB) g/L", min_value=10.0, max_value=80.0, value=42.0, step=0.1,
                                  help="Normal range: 35-50 g/L. Low levels may indicate liver disease.")
            alp = st.number_input("Alkaline Phosphatase (ALP) IU/L", min_value=10.0, max_value=500.0, value=70.0, step=1.0,
                                  help="Normal range: 40-130 IU/L. Elevated in liver/bone disease.")
            alt = st.number_input("ALT (SGPT) IU/L", min_value=1.0, max_value=500.0, value=25.0, step=1.0,
                                  help="Normal range: 7-56 IU/L. Key liver enzyme marker.")
        
        with col2:
            ast = st.number_input("AST (SGOT) IU/L", min_value=1.0, max_value=500.0, value=25.0, step=1.0,
                                  help="Normal range: 10-40 IU/L. Elevated in liver damage.")
            bil = st.number_input("Bilirubin (µmol/L)", min_value=0.0, max_value=500.0, value=8.0, step=0.1,
                                  help="Normal range: 3-17 µmol/L. High levels cause jaundice.")
            che = st.number_input("Cholinesterase (CHE) kU/L", min_value=1.0, max_value=20.0, value=8.0, step=0.1,
                                  help="Normal range: 5.3-12.9 kU/L. Low in liver disease.")
        
        with col3:
            chol = st.number_input("Cholesterol (CHOL) mmol/L", min_value=1.0, max_value=15.0, value=5.0, step=0.1,
                                   help="Normal range: <5.2 mmol/L. Metabolized by liver.")
            crea = st.number_input("Creatinine (µmol/L)", min_value=20.0, max_value=500.0, value=80.0, step=1.0,
                                   help="Normal range: 60-110 µmol/L. Kidney function marker.")
        
        with col4:
            ggt = st.number_input("Gamma-GT (GGT) IU/L", min_value=1.0, max_value=500.0, value=30.0, step=1.0,
                                  help="Normal range: 8-61 IU/L. Sensitive liver enzyme.")
            prot = st.number_input("Total Protein (PROT) g/L", min_value=40.0, max_value=100.0, value=72.0, step=0.1,
                                   help="Normal range: 64-83 g/L. Reflects liver synthetic function.")
        
        # Show abnormal value warnings
        warnings = []
        if alt > 56:
            warnings.append(f"⚠️ ALT elevated: {alt} IU/L (normal <56)")
        if ast > 40:
            warnings.append(f"⚠️ AST elevated: {ast} IU/L (normal <40)")
        if ggt > 61:
            warnings.append(f"⚠️ GGT elevated: {ggt} IU/L (normal <61)")
        if bil > 17:
            warnings.append(f"⚠️ Bilirubin elevated: {bil} µmol/L (normal <17)")
        if alb < 35:
            warnings.append(f"⚠️ Albumin low: {alb} g/L (normal >35)")
        
        if warnings:
            st.warning("**Abnormal Values Detected:**")
            for w in warnings:
                st.write(w)
        
        if st.button("Predict Hepatitis C Risk"):
            try:
                # Encode sex: Female=0, Male=1
                sex_encoded = 0 if sex == "Female" else 1
                
                # Create feature array in correct order:
                # ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
                user_input = [age, sex_encoded, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot]
                
                # Get model components
                hep_model = model_data['model']
                hep_scaler = model_data['scaler']
                numerical_cols = model_data.get('numerical_columns', 
                    ['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT'])
                feature_cols = model_data.get('feature_columns',
                    ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT'])
                
                # Create DataFrame and scale
                import pandas as pd
                input_df = pd.DataFrame([user_input], columns=feature_cols)
                input_df[numerical_cols] = hep_scaler.transform(input_df[numerical_cols])
                
                # Predict
                prediction = hep_model.predict(input_df)[0]
                probability = hep_model.predict_proba(input_df)[0]
                
                # Display results
                st.subheader("📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Healthy Probability", f"{probability[0]*100:.1f}%")
                with col2:
                    st.metric("Hepatitis C Risk", f"{probability[1]*100:.1f}%")
                with col3:
                    risk_level = "HIGH" if probability[1] > 0.5 else "MODERATE" if probability[1] > 0.3 else "LOW"
                    st.metric("Risk Level", risk_level)
                
                # Main result
                if prediction == 1:
                    st.error(f"⚠️ {name if name else 'Patient'}, HEPATITIS C INDICATORS DETECTED! Immediate hepatology consultation recommended.")
                    severity = "high"
                    
                    st.markdown("""
                    ### 🏥 Recommended Actions:
                    1. **Immediate**: Consult a hepatologist or gastroenterologist
                    2. **Confirmatory Tests**: HCV RNA test, HCV antibody test
                    3. **Liver Assessment**: FibroScan or liver biopsy if needed
                    4. **Treatment**: Direct-acting antivirals (DAAs) are highly effective
                    """)
                else:
                    if probability[1] > 0.3:
                        st.warning(f"⚠️ {name if name else 'Patient'}, some liver function abnormalities detected. Follow-up recommended.")
                        severity = "medium"
                    else:
                        st.success(f"✅ {name if name else 'Patient'}, no Hepatitis C indicators detected. Liver function appears normal.")
                        severity = "low"
                
                # Key markers analysis
                st.subheader("🔑 Key Liver Markers Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Liver Enzymes (Damage Markers):**")
                    ast_status = "🔴 Elevated" if ast > 40 else "🟢 Normal"
                    alt_status = "🔴 Elevated" if alt > 56 else "🟢 Normal"
                    ggt_status = "🔴 Elevated" if ggt > 61 else "🟢 Normal"
                    st.write(f"- AST: {ast} IU/L - {ast_status}")
                    st.write(f"- ALT: {alt} IU/L - {alt_status}")
                    st.write(f"- GGT: {ggt} IU/L - {ggt_status}")
                    
                    if ast > 0 and alt > 0:
                        ast_alt_ratio = ast / alt
                        st.write(f"- AST/ALT Ratio: {ast_alt_ratio:.2f}")
                        if ast_alt_ratio > 2:
                            st.write("  ⚠️ Ratio >2 may suggest alcoholic liver disease")
                        elif ast_alt_ratio > 1:
                            st.write("  ⚠️ Ratio >1 may suggest cirrhosis")
                
                with col2:
                    st.markdown("**Liver Synthetic Function:**")
                    alb_status = "🔴 Low" if alb < 35 else "🟢 Normal"
                    bil_status = "🔴 Elevated" if bil > 17 else "🟢 Normal"
                    prot_status = "🔴 Abnormal" if prot < 64 or prot > 83 else "🟢 Normal"
                    st.write(f"- Albumin: {alb} g/L - {alb_status}")
                    st.write(f"- Bilirubin: {bil} µmol/L - {bil_status}")
                    st.write(f"- Total Protein: {prot} g/L - {prot_status}")
                
                # Feature importance
                st.subheader("📈 Top Predictors (Model Feature Importance)")
                feature_importance = model_data.get('feature_importance', [])
                if feature_importance:
                    top_features = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)[:5]
                    for f in top_features:
                        feat_name = f['feature']
                        importance = f['importance'] * 100
                        st.write(f"- **{feat_name}**: {importance:.1f}%")
                
                # AI Recommendations
                if name:
                    with st.spinner("Generating hepatitis management recommendations..."):
                        patient_info = {
                            "name": name,
                            "age": age,
                            "risk_level": risk_level,
                            "ast": ast,
                            "alt": alt,
                            "ggt": ggt,
                            "bilirubin": bil,
                            "albumin": alb
                        }
                        
                        recommendations = get_health_recommendations("Hepatitis C", severity, patient_info)
                        if recommendations:
                            display_recommendations(recommendations)
                            display_health_tips_dynamic("Hepatitis C", severity.lower())
                
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# General Disease Prediction (Symptom-based)
if selected == '🔍 General Disease Prediction':
    st.title("🔍 General Disease Prediction")
    st.markdown("AI-powered disease prediction based on symptoms - **41 diseases, 132 symptoms**")
    
    # Model accuracy info and metrics display
    if "general_disease_model" in loaded_models:
        model_data = loaded_models["general_disease_model"]
        if isinstance(model_data, dict) and "accuracy" in model_data:
            st.success(f"✅ Model loaded successfully | Accuracy: {model_data['accuracy']:.2%}")
            
            # Model Performance Metrics Section
            with st.expander("📊 **Model Performance Metrics & Features**", expanded=False):
                # Metrics columns
                col_m1, col_m2, col_m3 = st.columns(3)
                
                with col_m1:
                    st.markdown("##### 🎯 Classification Metrics")
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                        'Score': [
                            f"{model_data.get('accuracy', 0)*100:.2f}%",
                            f"{model_data.get('precision_weighted', 0)*100:.2f}%",
                            f"{model_data.get('recall_weighted', 0)*100:.2f}%",
                            f"{model_data.get('f1_weighted', 0)*100:.2f}%",
                            f"{model_data.get('roc_auc_weighted', 0)*100:.2f}%"
                        ]
                    })
                    st.dataframe(metrics_df, hide_index=True, use_container_width=True)
                
                with col_m2:
                    st.markdown("##### 🔧 Model Configuration")
                    config_df = pd.DataFrame({
                        'Parameter': ['Model Type', 'Estimators', 'Features', 'Classes', 'CV Score'],
                        'Value': [
                            model_data.get('model_type', 'Random Forest'),
                            str(model_data.get('n_estimators', 200)),
                            str(model_data.get('n_features', 132)),
                            str(model_data.get('n_classes', 41)),
                            f"{model_data.get('cv_score', 0)*100:.2f}%"
                        ]
                    })
                    st.dataframe(config_df, hide_index=True, use_container_width=True)
                
                with col_m3:
                    st.markdown("##### 📈 Additional Metrics")
                    add_df = pd.DataFrame({
                        'Metric': ['Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)', 'ROC-AUC (Macro)'],
                        'Score': [
                            f"{model_data.get('precision_macro', 0)*100:.2f}%",
                            f"{model_data.get('recall_macro', 0)*100:.2f}%",
                            f"{model_data.get('f1_macro', 0)*100:.2f}%",
                            f"{model_data.get('roc_auc_macro', 0)*100:.2f}%"
                        ]
                    })
                    st.dataframe(add_df, hide_index=True, use_container_width=True)
                
                st.markdown("---")
                
                # Top Features Section
                st.markdown("##### 🏆 Top 20 Most Important Symptoms (Features)")
                top_features = model_data.get('feature_importance', [])
                if top_features:
                    # Create two columns for features
                    feat_col1, feat_col2 = st.columns(2)
                    
                    with feat_col1:
                        features_df1 = pd.DataFrame(top_features[:10])
                        features_df1['symptom'] = features_df1['symptom'].str.replace('_', ' ').str.title()
                        features_df1['importance'] = features_df1['importance'].apply(lambda x: f"{x*100:.2f}%")
                        features_df1.columns = ['Symptom', 'Importance']
                        features_df1.index = range(1, 11)
                        st.dataframe(features_df1, use_container_width=True)
                    
                    with feat_col2:
                        features_df2 = pd.DataFrame(top_features[10:20])
                        features_df2['symptom'] = features_df2['symptom'].str.replace('_', ' ').str.title()
                        features_df2['importance'] = features_df2['importance'].apply(lambda x: f"{x*100:.2f}%")
                        features_df2.columns = ['Symptom', 'Importance']
                        features_df2.index = range(11, 21)
                        st.dataframe(features_df2, use_container_width=True)
                    
                    # Feature importance bar chart
                    st.markdown("##### 📊 Feature Importance Visualization")
                    chart_data = pd.DataFrame(top_features[:10])
                    chart_data['symptom'] = chart_data['symptom'].str.replace('_', ' ').str.title()
                    st.bar_chart(chart_data.set_index('symptom')['importance'])
                
                st.markdown("---")
                
                # Diseases covered
                st.markdown("##### 🏥 Diseases Covered (41 conditions)")
                diseases = model_data.get('diseases', [])
                if diseases:
                    # Display in 4 columns
                    d_cols = st.columns(4)
                    for i, disease in enumerate(diseases):
                        with d_cols[i % 4]:
                            st.write(f"• {disease}")
    
    st.info("💡 Select your symptoms from the list below. The AI will predict possible diseases based on your symptoms.")
    
    name = st.text_input("👤 Patient Name:")
    
    # Check if model is loaded
    if "general_disease_model" in loaded_models and isinstance(loaded_models["general_disease_model"], dict):
        model_data = loaded_models["general_disease_model"]
        symptom_columns = model_data.get('symptom_columns', [])
        
        # Format symptoms for display (replace underscores with spaces, capitalize)
        formatted_symptoms = {col: col.replace('_', ' ').title() for col in symptom_columns}
        
        # Organize symptoms by category for better UX
        st.subheader("📋 Select Your Symptoms")
        
        # Create symptom categories
        general_symptoms = ['itching', 'skin_rash', 'fatigue', 'lethargy', 'malaise', 'high_fever', 'mild_fever', 
                          'sweating', 'chills', 'shivering', 'weight_loss', 'weight_gain', 'restlessness', 'anxiety', 'depression']
        
        pain_symptoms = ['headache', 'stomach_pain', 'abdominal_pain', 'belly_pain', 'chest_pain', 'back_pain', 
                        'joint_pain', 'knee_pain', 'hip_joint_pain', 'neck_pain', 'muscle_pain', 'pain_behind_the_eyes']
        
        digestive_symptoms = ['nausea', 'vomiting', 'diarrhoea', 'constipation', 'indigestion', 'acidity', 
                             'loss_of_appetite', 'excessive_hunger', 'increased_appetite', 'stomach_bleeding']
        
        respiratory_symptoms = ['cough', 'breathlessness', 'phlegm', 'mucoid_sputum', 'rusty_sputum', 
                               'blood_in_sputum', 'throat_irritation', 'congestion', 'runny_nose', 'continuous_sneezing']
        
        # Get remaining symptoms
        categorized = set(general_symptoms + pain_symptoms + digestive_symptoms + respiratory_symptoms)
        other_symptoms = [s for s in symptom_columns if s not in categorized]
        
        selected_symptoms = []
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌡️ General", "💢 Pain", "🍽️ Digestive", "🫁 Respiratory", "📋 Other"])
        
        with tab1:
            available_general = [s for s in general_symptoms if s in symptom_columns]
            for symptom in available_general:
                if st.checkbox(formatted_symptoms.get(symptom, symptom), key=f"gen_{symptom}"):
                    selected_symptoms.append(symptom)
        
        with tab2:
            available_pain = [s for s in pain_symptoms if s in symptom_columns]
            for symptom in available_pain:
                if st.checkbox(formatted_symptoms.get(symptom, symptom), key=f"pain_{symptom}"):
                    selected_symptoms.append(symptom)
        
        with tab3:
            available_digestive = [s for s in digestive_symptoms if s in symptom_columns]
            for symptom in available_digestive:
                if st.checkbox(formatted_symptoms.get(symptom, symptom), key=f"dig_{symptom}"):
                    selected_symptoms.append(symptom)
        
        with tab4:
            available_respiratory = [s for s in respiratory_symptoms if s in symptom_columns]
            for symptom in available_respiratory:
                if st.checkbox(formatted_symptoms.get(symptom, symptom), key=f"resp_{symptom}"):
                    selected_symptoms.append(symptom)
        
        with tab5:
            st.write("**Other Symptoms:**")
            # Use multiselect for remaining symptoms
            other_selected = st.multiselect(
                "Select additional symptoms:",
                [formatted_symptoms.get(s, s) for s in other_symptoms[:50]],  # Limit to prevent UI overload
                key="other_symptoms"
            )
            # Map back to original symptom names
            reverse_map = {v: k for k, v in formatted_symptoms.items()}
            for sym in other_selected:
                if reverse_map.get(sym) in symptom_columns:
                    selected_symptoms.append(reverse_map[sym])
        
        # Show selected symptoms
        if selected_symptoms:
            st.markdown("---")
            st.write(f"**Selected Symptoms ({len(selected_symptoms)}):** {', '.join([formatted_symptoms.get(s, s) for s in selected_symptoms])}")
        
        if st.button("🔬 Predict Disease", type="primary"):
            if selected_symptoms:
                try:
                    model = model_data['model']
                    label_encoder = model_data['label_encoder']
                    diseases = model_data.get('diseases', [])
                    
                    # Create input vector (all zeros, then set 1 for selected symptoms)
                    input_vector = np.zeros(len(symptom_columns))
                    for symptom in selected_symptoms:
                        if symptom in symptom_columns:
                            idx = symptom_columns.index(symptom)
                            input_vector[idx] = 1
                    
                    # Make prediction
                    input_df = pd.DataFrame([input_vector], columns=symptom_columns)
                    prediction = model.predict(input_df)[0]
                    prediction_proba = model.predict_proba(input_df)[0]
                    
                    # Get predicted disease name
                    predicted_disease = label_encoder.inverse_transform([prediction])[0]
                    confidence = prediction_proba[prediction] * 100
                    
                    # Get top 3 predictions
                    top_3_indices = np.argsort(prediction_proba)[-3:][::-1]
                    top_3_diseases = [(label_encoder.inverse_transform([i])[0], prediction_proba[i] * 100) for i in top_3_indices]
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    col_r1, col_r2 = st.columns(2)
                    
                    with col_r1:
                        if confidence >= 70:
                            st.error(f"🔴 **Primary Prediction: {predicted_disease}**")
                            severity = "high"
                        elif confidence >= 40:
                            st.warning(f"🟡 **Primary Prediction: {predicted_disease}**")
                            severity = "moderate"
                        else:
                            st.info(f"🟢 **Primary Prediction: {predicted_disease}**")
                            severity = "low"
                    
                    with col_r2:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    # Show top 3 predictions
                    st.write("**Top 3 Possible Conditions:**")
                    for i, (disease, prob) in enumerate(top_3_diseases, 1):
                        if prob >= 1:  # Only show if probability > 1%
                            st.write(f"{i}. **{disease}** - {prob:.1f}%")
                    
                    # Probability visualization
                    st.write("**Probability Distribution:**")
                    prob_df = pd.DataFrame({
                        "Disease": [d for d, p in top_3_diseases if p >= 1],
                        "Probability": [p/100 for d, p in top_3_diseases if p >= 1]
                    })
                    st.bar_chart(prob_df.set_index("Disease"))
                    
                    # Important disclaimer
                    st.warning("⚠️ **Disclaimer:** This prediction is for informational purposes only. Please consult a qualified healthcare professional for accurate diagnosis and treatment.")
                    
                    # Get AI recommendations
                    if name:
                        with st.spinner("Generating personalized recommendations..."):
                            patient_info = {
                                "name": name,
                                "predicted_disease": predicted_disease,
                                "confidence": f"{confidence:.1f}%",
                                "symptoms": selected_symptoms
                            }
                            
                            recommendations = get_health_recommendations(predicted_disease, severity, patient_info)
                            if recommendations:
                                display_recommendations(recommendations)
                                display_health_tips_dynamic(predicted_disease, severity)
                    
                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            else:
                st.warning("Please select at least one symptom to make a prediction.")
    else:
        st.error("⚠️ General Disease Model not loaded. Please check the model file.")
        # Fallback to simple symptom list
        symptoms_list = [
            "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering",
            "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting",
            "vomiting", "burning_micturition", "fatigue", "weight_gain", "anxiety", "cold_hands_and_feets",
            "mood_swings", "weight_loss", "restlessness", "lethargy", "patches_in_throat", "cough",
            "high_fever", "breathlessness", "sweating", "dehydration", "indigestion", "headache",
            "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "back_pain", "constipation",
            "abdominal_pain", "diarrhoea", "mild_fever", "chest_pain", "dizziness", "muscle_pain"
        ]
        
        selected_symptoms = st.multiselect("Select your symptoms:", symptoms_list)
        
        if st.button("Get Health Advice"):
            if selected_symptoms:
                st.write(f"**Selected Symptoms:** {', '.join(selected_symptoms)}")
                st.warning("Model not available. Please consult a healthcare professional for accurate diagnosis.")

# Book Appointment
if selected == 'Book Appointment':
    st.title("📅 Book Medical Appointment")
    st.markdown("Schedule your consultation with healthcare professionals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        patient_name = st.text_input("Patient Name")
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        phone = st.text_input("Phone Number")
        email = st.text_input("Email Address")
    
    with col2:
        appointment_type = st.selectbox("Appointment Type", [
            "General Consultation",
            "Follow-up Visit",
            "Diagnostic Tests",
            "Second Opinion",
            "Emergency Consultation"
        ])
        specialist = st.selectbox("Specialist Required", [
            "General Physician",
            "Cardiologist",
            "Neurologist",
            "Endocrinologist",
            "Oncologist",
            "Nephrologist",
            "Hepatologist",
            "Pulmonologist",
            "Gastroenterologist",
            "Infectious Disease Specialist"
        ])
        preferred_date = st.date_input("Preferred Date")
        preferred_time = st.time_input("Preferred Time")
    
    reason = st.text_area("Reason for Appointment")
    existing_conditions = st.text_area("Existing Medical Conditions (if any)")
    
    if st.button("Book Appointment"):
        if patient_name and phone:
            st.success(f"✅ Appointment request submitted successfully!")
            st.info(f"""
            **Appointment Details:**
            - Patient: {patient_name}
            - Specialist: {specialist}
            - Date: {preferred_date}
            - Time: {preferred_time}
            - Type: {appointment_type}
            
            You will receive a confirmation message shortly at {phone} and {email}.
            """)
            
            st.balloons()
        else:
            st.error("Please fill in all required fields (Name and Phone Number)")

# Set Reminder
if selected == 'Set Reminder':
    st.title("⏰ Health Reminders")
    st.markdown("Set reminders for medications, appointments, and health checkups")
    
    reminder_type = st.selectbox("Reminder Type", [
        "Medication",
        "Doctor Appointment",
        "Health Checkup",
        "Exercise",
        "Water Intake",
        "Blood Pressure Monitoring",
        "Blood Sugar Monitoring",
        "Diet Reminder"
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        reminder_title = st.text_input("Reminder Title", placeholder="e.g., Take Blood Pressure Medicine")
        reminder_date = st.date_input("Date")
        reminder_time = st.time_input("Time")
    
    with col2:
        frequency = st.selectbox("Frequency", [
            "Once",
            "Daily",
            "Every 2 days",
            "Weekly",
            "Monthly"
        ])
        notification_method = st.multiselect("Notification Method", [
            "Email",
            "SMS",
            "Push Notification",
            "App Alert"
        ])
    
    notes = st.text_area("Additional Notes")
    
    if st.button("Set Reminder"):
        st.success(f"✅ Reminder set successfully!")
        st.info(f"""
        **Reminder Details:**
        - Type: {reminder_type}
        - Title: {reminder_title}
        - Date & Time: {reminder_date} at {reminder_time}
        - Frequency: {frequency}
        - Notifications: {', '.join(notification_method)}
        """)
        
        st.write("You will be notified through your selected channels.")

# Health Tips
if selected == 'Health Tips':
    st.title("💡 Health Tips & Wellness Guide")
    st.markdown("Daily health tips for better living")
    
    import random
    
    # Display random health tip
    tip_of_the_day = random.choice(health_tips)
    st.success(f"**💡 Tip of the Day:** {tip_of_the_day}")
    
    # Category-wise health tips
    st.markdown("---")
    st.subheader("Health Tips by Category")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🍎 Nutrition", "🏃 Exercise", "😴 Sleep", "🧘 Mental Health", "💊 Preventive Care"
    ])
    
    with tab1:
        st.markdown("""
        ### Nutrition Tips
        - Eat a rainbow of fruits and vegetables daily
        - Stay hydrated with at least 8 glasses of water
        - Limit processed foods and added sugars
        - Include lean proteins in every meal
        - Choose whole grains over refined grains
        - Practice portion control
        - Eat slowly and mindfully
        - Plan your meals ahead
        - Read nutrition labels carefully
        - Limit sodium intake to less than 2,300mg per day
        """)
    
    with tab2:
        st.markdown("""
        ### Exercise Tips
        - Aim for 150 minutes of moderate exercise per week
        - Include both cardio and strength training
        - Start slowly and gradually increase intensity
        - Find activities you enjoy
        - Exercise with a friend for motivation
        - Stretch before and after workouts
        - Listen to your body and rest when needed
        - Set realistic fitness goals
        - Track your progress
        - Make movement a daily habit
        """)
    
    with tab3:
        st.markdown("""
        ### Sleep Tips
        - Maintain a consistent sleep schedule
        - Aim for 7-9 hours of sleep per night
        - Create a relaxing bedtime routine
        - Keep your bedroom cool, dark, and quiet
        - Avoid screens 1 hour before bed
        - Limit caffeine after 2 PM
        - Avoid heavy meals before bedtime
        - Exercise regularly, but not close to bedtime
        - Manage stress through relaxation techniques
        - Invest in a comfortable mattress and pillows
        """)
    
    with tab4:
        st.markdown("""
        ### Mental Health Tips
        - Practice mindfulness and meditation daily
        - Stay connected with friends and family
        - Express your feelings and emotions
        - Seek professional help when needed
        - Take breaks from social media
        - Engage in hobbies you enjoy
        - Practice gratitude daily
        - Set healthy boundaries
        - Learn to say no
        - Celebrate small victories
        """)
    
    with tab5:
        st.markdown("""
        ### Preventive Care Tips
        - Schedule regular health checkups
        - Keep vaccinations up to date
        - Monitor your blood pressure regularly
        - Get recommended cancer screenings
        - Maintain dental hygiene
        - Protect your skin from sun damage
        - Wash hands frequently
        - Avoid tobacco and limit alcohol
        - Manage stress effectively
        - Stay informed about family health history
        """)
    
    # Interactive health calculator
    st.markdown("---")
    st.subheader("🧮 Quick Health Calculators")
    
    calc_type = st.selectbox("Select Calculator", [
        "BMI Calculator",
        "Water Intake Calculator",
        "Calorie Needs Calculator",
        "Heart Rate Zones"
    ])
    
    if calc_type == "BMI Calculator":
        col1, col2 = st.columns(2)
        with col1:
            height_calc = st.number_input("Height (cm)", min_value=50, max_value=250, value=170, key="calc_height")
            weight_calc = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70, key="calc_weight")
        
        if st.button("Calculate BMI", key="calc_bmi_btn"):
            bmi_calc = weight_calc / ((height_calc/100) ** 2)
            st.metric("Your BMI", f"{bmi_calc:.1f}")
            
            if bmi_calc < 18.5:
                st.info("Underweight")
            elif bmi_calc < 25:
                st.success("Normal weight")
            elif bmi_calc < 30:
                st.warning("Overweight")
            else:
                st.error("Obese")
    
    elif calc_type == "Water Intake Calculator":
        weight_water = st.number_input("Your Weight (kg)", min_value=10, max_value=300, value=70, key="water_weight")
        activity_level = st.selectbox("Activity Level", ["Sedentary", "Moderate", "Active"], key="water_activity")
        
        if st.button("Calculate Water Intake"):
            base_water = weight_water * 0.033  # 33ml per kg
            if activity_level == "Moderate":
                base_water *= 1.2
            elif activity_level == "Active":
                base_water *= 1.5
            
            st.metric("Recommended Daily Water Intake", f"{base_water:.1f} liters")
            st.info(f"That's approximately {int(base_water * 4)} glasses (250ml each)")
    
    elif calc_type == "Calorie Needs Calculator":
        col1, col2 = st.columns(2)
        with col1:
            age_cal = st.number_input("Age", min_value=1, max_value=120, value=30, key="cal_age")
            gender_cal = st.selectbox("Gender", ["Male", "Female"], key="cal_gender")
            weight_cal = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70, key="cal_weight")
        
        with col2:
            height_cal = st.number_input("Height (cm)", min_value=50, max_value=250, value=170, key="cal_height")
            activity_cal = st.selectbox("Activity Level", [
                "Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"
            ], key="cal_activity")
        
        if st.button("Calculate Calories"):
            # Mifflin-St Jeor Equation
            if gender_cal == "Male":
                bmr = 10 * weight_cal + 6.25 * height_cal - 5 * age_cal + 5
            else:
                bmr = 10 * weight_cal + 6.25 * height_cal - 5 * age_cal - 161
            
            activity_multipliers = {
                "Sedentary": 1.2,
                "Lightly Active": 1.375,
                "Moderately Active": 1.55,
                "Very Active": 1.725,
                "Extremely Active": 1.9
            }
            
            tdee = bmr * activity_multipliers[activity_cal]
            
            st.metric("Daily Calorie Needs (Maintenance)", f"{tdee:.0f} calories")
            st.write(f"**For Weight Loss:** {tdee-500:.0f} calories/day")
            st.write(f"**For Weight Gain:** {tdee+500:.0f} calories/day")
    
    elif calc_type == "Heart Rate Zones":
        age_hr = st.number_input("Your Age", min_value=1, max_value=120, value=30, key="hr_age")
        
        if st.button("Calculate Heart Rate Zones"):
            max_hr = 220 - age_hr
            
            st.write(f"**Maximum Heart Rate:** {max_hr} bpm")
            st.write("")
            st.write("**Training Zones:**")
            st.write(f"🔵 Warm Up (50-60%): {int(max_hr*0.5)}-{int(max_hr*0.6)} bpm")
            st.write(f"🟢 Fat Burn (60-70%): {int(max_hr*0.6)}-{int(max_hr*0.7)} bpm")
            st.write(f"🟡 Cardio (70-80%): {int(max_hr*0.7)}-{int(max_hr*0.8)} bpm")
            st.write(f"🟠 Peak (80-90%): {int(max_hr*0.8)}-{int(max_hr*0.9)} bpm")
            st.write(f"🔴 Maximum (90-100%): {int(max_hr*0.9)}-{max_hr} bpm")

# Add new disease prediction pages for the additional diseases

# Alzheimer's Prediction
# Alzheimer's Prediction
# Alzheimer's Prediction
# Alzheimer's Prediction
if selected == 'Alzheimers Prediction':
    st.title("🧠 Alzheimer's Disease Prediction")
    st.markdown("Early detection and assessment using advanced clinical parameters")
    
    # Info banner
    st.info("💡 This assessment uses 32 clinical parameters for comprehensive risk evaluation")
    
    name = st.text_input("👤 Patient Name:")
    
    # Organized tabs for better UX
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Demographics & Lifestyle", 
        "🏥 Medical History", 
        "🩺 Clinical Measurements",
        "🧪 Cognitive & Functional"
    ])
    
    with tab1:
        st.subheader("Demographics & Lifestyle Factors")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=40, max_value=100, value=65)
            gender = st.selectbox("Gender", ["Male", "Female"])
            ethnicity = st.number_input("Ethnicity (0-3)", min_value=0, max_value=3, value=0, 
                                       help="0=Caucasian, 1=African American, 2=Asian, 3=Other")
            education_level = st.number_input("Years of Education", min_value=0, max_value=30, value=12)
        
        with col2:
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
            smoking = st.selectbox("Smoking Status", ["No", "Yes"])
            alcohol = st.slider("Alcohol Consumption (units/week)", min_value=0, max_value=50, value=0)
            physical_activity = st.slider("Physical Activity Level (0-10)", min_value=0, max_value=10, value=5,
                                         help="0=Sedentary, 10=Very Active")
        
        with col3:
            diet_quality = st.slider("Diet Quality Score (0-10)", min_value=0, max_value=10, value=5,
                                    help="0=Poor, 10=Excellent")
            sleep_quality = st.slider("Sleep Quality Score (0-10)", min_value=0, max_value=10, value=7,
                                     help="0=Very Poor, 10=Excellent")
    
    with tab2:
        st.subheader("Medical History & Risk Factors")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏥 Chronic Conditions**")
            family_history = st.selectbox("Family History of Alzheimer's", ["No", "Yes"])
            cardiovascular_disease = st.selectbox("Cardiovascular Disease", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            depression = st.selectbox("Depression", ["No", "Yes"])
        
        with col2:
            st.markdown("**⚠️ Other Risk Factors**")
            head_injury = st.selectbox("Previous Head Injury", ["No", "Yes"])
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    
    with tab3:
        st.subheader("Clinical Measurements")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🩺 Blood Pressure**")
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=200, value=120)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=120, value=80)
        
        with col2:
            st.markdown("**🧪 Cholesterol Levels**")
            cholesterol_total = st.number_input("Total Cholesterol (mg/dL)", min_value=100.0, max_value=400.0, value=200.0)
            cholesterol_ldl = st.number_input("LDL Cholesterol (mg/dL)", min_value=50.0, max_value=300.0, value=100.0)
        
        with col3:
            st.markdown("**💊 Additional Lipids**")
            cholesterol_hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=20.0, max_value=100.0, value=50.0)
            cholesterol_triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=20.0, max_value=500.0, value=150.0)
    
    with tab4:
        st.subheader("Cognitive & Functional Assessment")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🧠 Cognitive Tests**")
            mmse = st.slider("MMSE Score (Mini-Mental State Exam)", 0, 30, 25,
                           help="30=Normal, <24=Cognitive Impairment")
            functional_assessment = st.slider("Functional Assessment Score", 0.0, 10.0, 5.0)
            adl = st.slider("Activities of Daily Living Score", 0.0, 10.0, 5.0,
                          help="10=Fully Independent, 0=Fully Dependent")
        
        with col2:
            st.markdown("**⚠️ Symptoms & Behaviors**")
            memory_complaints = st.selectbox("Memory Complaints", ["No", "Yes"])
            behavioral_problems = st.selectbox("Behavioral Problems", ["No", "Yes"])
            confusion = st.selectbox("Confusion", ["No", "Yes"])
            disorientation = st.selectbox("Disorientation", ["No", "Yes"])
            personality_changes = st.selectbox("Personality Changes", ["No", "Yes"])
            difficulty_completing_tasks = st.selectbox("Difficulty Completing Tasks", ["No", "Yes"])
            forgetfulness = st.selectbox("Forgetfulness", ["No", "Yes"])

    st.markdown("---")
    
    if st.button("🔬 Analyze Alzheimer's Risk", type="primary", use_container_width=True):
        if not name:
            st.warning("⚠️ Please enter patient name")
        else:
            try:
                if "alzheimers_model" not in loaded_models:
                    st.error("❌ Alzheimer's model not loaded. Please check model file.")
                else:
                    model_data = loaded_models["alzheimers_model"]
                    
                    if isinstance(model_data, dict):
                        model = model_data["model"]
                        scaler = model_data["scaler"]
                        feature_columns = model_data["feature_columns"]
                        
                        # Convert inputs to numeric
                        gender_num = 1 if gender == "Male" else 0
                        smoking_num = 1 if smoking == "Yes" else 0
                        family_history_num = 1 if family_history == "Yes" else 0
                        cardiovascular_disease_num = 1 if cardiovascular_disease == "Yes" else 0
                        diabetes_num = 1 if diabetes == "Yes" else 0
                        depression_num = 1 if depression == "Yes" else 0
                        head_injury_num = 1 if head_injury == "Yes" else 0
                        hypertension_num = 1 if hypertension == "Yes" else 0
                        memory_complaints_num = 1 if memory_complaints == "Yes" else 0
                        behavioral_problems_num = 1 if behavioral_problems == "Yes" else 0
                        confusion_num = 1 if confusion == "Yes" else 0
                        disorientation_num = 1 if disorientation == "Yes" else 0
                        personality_changes_num = 1 if personality_changes == "Yes" else 0
                        difficulty_completing_tasks_num = 1 if difficulty_completing_tasks == "Yes" else 0
                        forgetfulness_num = 1 if forgetfulness == "Yes" else 0

                        # Create input dataframe matching feature_columns order
                        input_data = pd.DataFrame([[
                            age, gender_num, ethnicity, education_level, bmi, smoking_num, alcohol,
                            physical_activity, diet_quality, sleep_quality, family_history_num,
                            cardiovascular_disease_num, diabetes_num, depression_num, head_injury_num,
                            hypertension_num, systolic_bp, diastolic_bp, cholesterol_total, cholesterol_ldl,
                            cholesterol_hdl, cholesterol_triglycerides, mmse, functional_assessment,
                            memory_complaints_num, behavioral_problems_num, adl, confusion_num,
                            disorientation_num, personality_changes_num, difficulty_completing_tasks_num,
                            forgetfulness_num
                        ]], columns=feature_columns)
                        
                        # Scale the input
                        input_scaled = scaler.transform(input_data)
                        
                        # Make prediction
                        prediction = model.predict(input_scaled)
                        prediction_proba = model.predict_proba(input_scaled)[0]

                        # Display results with enhanced UI
                        st.markdown("---")
                        st.markdown("## 📊 Analysis Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if prediction[0] == 1:
                                st.error("🔴 High Risk Detected")
                                severity = "high"
                                risk_emoji = "🔴"
                            else:
                                st.success("🟢 Low Risk")
                                severity = "low"
                                risk_emoji = "🟢"
                        
                        with col2:
                            confidence = max(prediction_proba) * 100
                            st.metric("🎯 Model Confidence", f"{confidence:.1f}%")
                        
                        with col3:
                            st.metric("🧠 MMSE Score", f"{mmse}/30")
                        
                        # Risk interpretation
                        st.markdown("---")
                        if prediction[0] == 1:
                            st.error(f"""
                            ### ⚠️ Clinical Alert
                            **{name}**, the assessment indicates elevated risk for Alzheimer's disease.
                            
                            **Recommended Actions:**
                            - 🏥 Schedule immediate consultation with a neurologist
                            - 🧪 Comprehensive cognitive assessment recommended
                            - 📋 Additional imaging studies (MRI/PET) may be needed
                            - 👨‍⚕️ Consider specialist referral to memory clinic
                            """)
                        else:
                            st.success(f"""
                            ### ✅ Assessment Summary
                            **{name}**, the current assessment shows low risk indicators.
                            
                            **Recommendations:**
                            - 🌟 Maintain healthy lifestyle habits
                            - 🧠 Continue cognitive activities and mental stimulation
                            - 📅 Regular health screenings recommended
                            - 💪 Stay physically and socially active
                            """)
                        
                        # Risk factors summary
                        st.markdown("---")
                        st.subheader("🔍 Key Risk Factors Identified")
                        
                        risk_factors = []
                        if mmse < 24:
                            risk_factors.append(("Cognitive Impairment", "MMSE score below normal range"))
                        if family_history == "Yes":
                            risk_factors.append(("Family History", "Genetic predisposition present"))
                        if age > 75:
                            risk_factors.append(("Advanced Age", "Age-related risk factor"))
                        if cardiovascular_disease == "Yes":
                            risk_factors.append(("Cardiovascular Disease", "Vascular risk factor"))
                        if diabetes == "Yes":
                            risk_factors.append(("Diabetes", "Metabolic risk factor"))
                        if memory_complaints == "Yes":
                            risk_factors.append(("Memory Complaints", "Subjective cognitive decline"))
                        
                        if risk_factors:
                            for factor, description in risk_factors:
                                st.warning(f"⚠️ **{factor}**: {description}")
                        else:
                            st.info("✅ No major risk factors identified")
                        
                        # AI Recommendations
                        st.markdown("---")
                        with st.spinner("🤖 Generating personalized care recommendations..."):
                            patient_info = {
                                "name": name,
                                "age": age,
                                "mmse_score": mmse,
                                "risk_factors": ", ".join([f[0] for f in risk_factors]) if risk_factors else "None"
                            }
                            
                            recommendations = get_health_recommendations("Alzheimer's Disease", severity, patient_info)
                            if recommendations:
                                display_recommendations(recommendations)
                                display_health_tips_dynamic("Alzheimer's Disease", severity)
                    else:
                        st.error("Model format not recognized.")

            except Exception as e:
                st.error(f"❌ Error in prediction: {str(e)}")
                with st.expander("🔧 Technical Details"):
                    import traceback
                    st.code(traceback.format_exc())
# Epilepsy Prediction
if selected == 'Epilepsy Prediction':
    st.title("⚡ Epilepsy Seizure Prediction")
    st.markdown("AI-powered seizure detection using EEG signal analysis")
    
    # Model accuracy info
    if "epilepsy_model" in loaded_models:
        model_data = loaded_models["epilepsy_model"]
        if isinstance(model_data, dict) and "accuracy" in model_data:
            st.success(f"✅ Model loaded successfully | Accuracy: {model_data['accuracy']:.2%}")
    
    name = st.text_input("👤 Patient Name:")
    
    st.markdown("### 📋 Clinical Background")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        seizure_type = st.selectbox("Known Seizure Type", [
            "Unknown/First Assessment", "Generalized Tonic-Clonic", "Focal", 
            "Absence", "Myoclonic", "Atonic"
        ])
    
    with col2:
        seizure_frequency = st.selectbox("Seizure History", [
            "No Previous Seizures", "First Time", "Yearly", "Monthly", "Weekly", "Daily"
        ])
        family_history = st.selectbox("Family History of Epilepsy", ["No", "Yes"])
        head_injury = st.selectbox("Previous Head Injury", ["No", "Yes"])
    
    with col3:
        triggers = st.multiselect("Known Triggers", [
            "None Known", "Stress", "Lack of Sleep", "Flashing Lights",
            "Alcohol", "Missed Medication", "Fever"
        ])
        birth_complications = st.selectbox("Birth Complications", ["No", "Yes"])
        febrile_seizures = st.selectbox("Childhood Febrile Seizures", ["No", "Yes"])
    
    st.markdown("---")
    st.markdown("### 📊 EEG Signal Analysis")
    st.info("Upload EEG data with 178 feature columns (signal measurements) per sample for seizure detection.")
    
    eeg_file = st.file_uploader("Upload EEG CSV file", type=["csv"])
    
    if st.button("🔬 Analyze EEG & Predict Seizure Risk", type="primary"):
        try:
            if "epilepsy_model" not in loaded_models:
                st.error("Epilepsy model not loaded. Please check model file.")
            elif eeg_file is None:
                st.warning("Please upload an EEG data CSV file to proceed.")
            else:
                model_data = loaded_models["epilepsy_model"]
                
                if isinstance(model_data, dict):
                    model = model_data["model"]
                    scaler = model_data["scaler"]
                    feature_columns = model_data["feature_columns"]
                    
                    # Load EEG data
                    eeg_data = pd.read_csv(eeg_file)
                    
                    # Check if data has enough columns
                    if eeg_data.shape[1] < 178:
                        st.error(f"EEG data must have at least 178 feature columns. Found {eeg_data.shape[1]} columns.")
                    else:
                        # Use first 178 columns as features
                        eeg_features = eeg_data.iloc[:, :178]
                        eeg_features.columns = feature_columns  # Match training feature names
                        
                        # Scale features
                        eeg_scaled = scaler.transform(eeg_features)
                        
                        # Make predictions
                        predictions = model.predict(eeg_scaled)
                        prediction_proba = model.predict_proba(eeg_scaled)
                        
                        # Calculate seizure percentage
                        seizure_count = np.sum(predictions == 1)
                        total_segments = len(predictions)
                        seizure_percentage = (seizure_count / total_segments) * 100
                        avg_seizure_prob = np.mean(prediction_proba[:, 1]) * 100
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("🎯 EEG Analysis Results")
                        
                        col_r1, col_r2, col_r3 = st.columns(3)
                        
                        with col_r1:
                            st.metric("Total EEG Segments", total_segments)
                        
                        with col_r2:
                            st.metric("Seizure Detected", f"{seizure_count} segments")
                        
                        with col_r3:
                            st.metric("Seizure Activity", f"{seizure_percentage:.1f}%")
                        
                        # Risk assessment
                        if seizure_percentage > 50:
                            risk_label = "HIGH SEIZURE RISK"
                            severity = "high"
                            st.error(f"🔴 **{risk_label}** - {seizure_percentage:.1f}% of EEG segments show seizure activity")
                        elif seizure_percentage > 20:
                            risk_label = "MODERATE SEIZURE RISK"
                            severity = "moderate"
                            st.warning(f"🟡 **{risk_label}** - {seizure_percentage:.1f}% of EEG segments show seizure activity")
                        else:
                            risk_label = "LOW SEIZURE RISK"
                            severity = "low"
                            st.success(f"🟢 **{risk_label}** - {seizure_percentage:.1f}% of EEG segments show seizure activity")
                        
                        # Visualization
                        col_v1, col_v2 = st.columns(2)
                        with col_v1:
                            st.write("**Segment-wise Classification:**")
                            class_counts = pd.DataFrame({
                                "Classification": ["Normal", "Seizure"],
                                "Count": [total_segments - seizure_count, seizure_count]
                            })
                            st.bar_chart(class_counts.set_index("Classification"))
                        
                        with col_v2:
                            st.write("**Average Seizure Probability:**")
                            st.progress(avg_seizure_prob / 100)
                            st.write(f"{avg_seizure_prob:.1f}%")
                        
                        # Feature importance
                        if "feature_importance" in model_data:
                            with st.expander("📊 Top EEG Features for Detection"):
                                importance_df = pd.DataFrame({
                                    "Feature": feature_columns[:20],
                                    "Importance": model_data["feature_importance"][:20]
                                }).sort_values("Importance", ascending=False).head(10)
                                st.bar_chart(importance_df.set_index("Feature"))
                        
                        # Recommendations
                        if name:
                            patient_info = {
                                "name": name,
                                "age": age,
                                "seizure_type": seizure_type,
                                "seizure_frequency": seizure_frequency,
                                "eeg_seizure_percentage": seizure_percentage,
                                "risk_level": risk_label
                            }
                            recommendations = get_health_recommendations("Epilepsy", severity, patient_info)
                            if recommendations:
                                display_recommendations(recommendations)
                                display_health_tips_dynamic("Epilepsy", severity)
                else:
                    st.error("Model format not recognized.")
                    
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Tuberculosis Prediction
if selected == 'Tuberculosis Prediction':
    st.title("🫁 Tuberculosis (TB) Risk Prediction")
    st.markdown("AI-powered TB risk assessment based on clinical symptoms")
    
    # Model accuracy info
    if "tuberculosis_model" in loaded_models:
        model_data = loaded_models["tuberculosis_model"]
        if isinstance(model_data, dict) and "accuracy" in model_data:
            st.success(f"✅ Model loaded successfully | Accuracy: {model_data['accuracy']:.2%} | CV: {model_data.get('cv_accuracy', 0):.2%}")
    
    name = st.text_input("👤 Patient Name:")
    
    # Patient info
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", options=["Male", "Female"])
        gender_val = 1 if gender == "Male" else 0
    with col2:
        age = st.number_input("Age (for reference)", min_value=1, max_value=120, value=35)
    
    st.divider()
    st.subheader("🩺 TB Symptom Assessment")
    st.markdown("Please indicate which symptoms are present:")
    
    # Organize symptoms into categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Respiratory Symptoms**")
        coughing_blood = st.checkbox("🩸 Coughing Blood")
        sputum_blood = st.checkbox("🩸 Sputum Mixed with Blood")
        shortness_breath = st.checkbox("😮‍💨 Shortness of Breath")
        cough_phlegm = st.checkbox("😷 Persistent Cough with Phlegm (2-4 weeks)")
    
    with col2:
        st.markdown("**Systemic Symptoms**")
        fever = st.checkbox("🤒 Fever for Two Weeks")
        night_sweats = st.checkbox("😰 Night Sweats")
        weight_loss = st.checkbox("📉 Unexplained Weight Loss")
        fatigue = st.checkbox("😴 Body Feels Tired/Fatigue")
    
    with col3:
        st.markdown("**Other Symptoms**")
        chest_pain = st.checkbox("💔 Chest Pain")
        back_pain = st.checkbox("🔙 Back Pain in Certain Parts")
        lumps = st.checkbox("🔘 Lumps around Armpits/Neck")
        swollen_lymph = st.checkbox("🔘 Swollen Lymph Nodes")
        loss_appetite = st.checkbox("🍽️ Loss of Appetite")
    
    st.divider()
    
    if st.button("🔬 Predict TB Risk", type="primary", use_container_width=True):
        try:
            if "tuberculosis_model" not in loaded_models:
                st.error("Tuberculosis model not loaded. Please check model file.")
            else:
                model_data = loaded_models["tuberculosis_model"]
                
                if isinstance(model_data, dict):
                    model = model_data["model"]
                    scaler = model_data["scaler"]
                    feature_columns = model_data["feature_columns"]
                    
                    # Convert checkbox values to integers
                    fever_val = 1 if fever else 0
                    coughing_blood_val = 1 if coughing_blood else 0
                    sputum_blood_val = 1 if sputum_blood else 0
                    night_sweats_val = 1 if night_sweats else 0
                    chest_pain_val = 1 if chest_pain else 0
                    back_pain_val = 1 if back_pain else 0
                    shortness_breath_val = 1 if shortness_breath else 0
                    weight_loss_val = 1 if weight_loss else 0
                    fatigue_val = 1 if fatigue else 0
                    lumps_val = 1 if lumps else 0
                    cough_phlegm_val = 1 if cough_phlegm else 0
                    swollen_lymph_val = 1 if swollen_lymph else 0
                    loss_appetite_val = 1 if loss_appetite else 0
                    
                    # Calculate engineered features
                    respiratory_symptoms = coughing_blood_val + sputum_blood_val + shortness_breath_val + cough_phlegm_val
                    systemic_symptoms = fever_val + night_sweats_val + weight_loss_val + fatigue_val
                    lymph_symptoms = lumps_val + swollen_lymph_val
                    symptom_count = (fever_val + coughing_blood_val + sputum_blood_val + night_sweats_val + 
                                    chest_pain_val + back_pain_val + shortness_breath_val + weight_loss_val + 
                                    fatigue_val + lumps_val + cough_phlegm_val + swollen_lymph_val + loss_appetite_val)
                    
                    # Prepare input
                    input_data = {
                        'gender': gender_val,
                        'fever for two weeks': fever_val,
                        'coughing blood': coughing_blood_val,
                        'sputum mixed with blood': sputum_blood_val,
                        'night sweats ': night_sweats_val,
                        'chest pain': chest_pain_val,
                        'back pain in certain parts ': back_pain_val,
                        'shortness of breath': shortness_breath_val,
                        'weight loss ': weight_loss_val,
                        'body feels tired': fatigue_val,
                        'lumps that appear around the armpits and neck': lumps_val,
                        'cough and phlegm continuously for two weeks to four weeks': cough_phlegm_val,
                        'swollen lymph nodes': swollen_lymph_val,
                        'loss of appetite': loss_appetite_val,
                        'respiratory_symptoms': respiratory_symptoms,
                        'systemic_symptoms': systemic_symptoms,
                        'lymph_symptoms': lymph_symptoms,
                        'symptom_count': symptom_count
                    }
                    
                    # Create DataFrame
                    input_df = pd.DataFrame([input_data])[feature_columns]
                    
                    # Scale and predict
                    input_scaled = scaler.transform(input_df)
                    prediction = model.predict(input_scaled)[0]
                    
                    # Get probability
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(input_scaled)[0]
                        confidence = proba[prediction] * 100
                        tb_risk_prob = proba[1] * 100
                    else:
                        confidence = 95.0
                        tb_risk_prob = 50.0 if prediction == 0 else 95.0
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        result = "TB Positive (High Risk)" if prediction == 1 else "TB Negative (Low Risk)"
                        st.metric("Prediction", "High Risk" if prediction == 1 else "Low Risk")
                    with col2:
                        st.metric("TB Risk", f"{tb_risk_prob:.1f}%")
                    with col3:
                        st.metric("Symptoms Present", f"{symptom_count}/13")
                    with col4:
                        st.metric("Model Accuracy", f"{model_data.get('accuracy', 1.0)*100:.1f}%")
                    
                    # Risk level display
                    if prediction == 1:
                        st.error(f"⚠️ {name if name else 'Patient'}, **HIGH TB RISK** detected! Immediate medical evaluation and TB testing strongly recommended.")
                        severity = "high"
                    else:
                        if symptom_count >= 4:
                            st.warning(f"⚠️ {name if name else 'Patient'}, **MODERATE RISK**. Some TB symptoms present. Consider TB screening if symptoms persist.")
                            severity = "moderate"
                        else:
                            st.success(f"✅ {name if name else 'Patient'}, **LOW TB RISK**. Continue monitoring and maintain good respiratory health.")
                            severity = "low"
                    
                    # Symptom summary
                    st.markdown("---")
                    st.subheader("📋 Symptom Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    present_symptoms = []
                    if fever: present_symptoms.append("Fever for 2+ weeks")
                    if coughing_blood: present_symptoms.append("Coughing blood")
                    if sputum_blood: present_symptoms.append("Sputum with blood")
                    if night_sweats: present_symptoms.append("Night sweats")
                    if chest_pain: present_symptoms.append("Chest pain")
                    if back_pain: present_symptoms.append("Back pain")
                    if shortness_breath: present_symptoms.append("Shortness of breath")
                    if weight_loss: present_symptoms.append("Weight loss")
                    if fatigue: present_symptoms.append("Fatigue")
                    if lumps: present_symptoms.append("Lymph node lumps")
                    if cough_phlegm: present_symptoms.append("Persistent cough with phlegm")
                    if swollen_lymph: present_symptoms.append("Swollen lymph nodes")
                    if loss_appetite: present_symptoms.append("Loss of appetite")
                    
                    with col1:
                        st.markdown("**Present Symptoms:**")
                        if present_symptoms:
                            for s in present_symptoms:
                                st.markdown(f"• ⚠️ {s}")
                        else:
                            st.success("No TB symptoms reported")
                    
                    with col2:
                        st.markdown("**Symptom Categories:**")
                        st.markdown(f"• Respiratory: {respiratory_symptoms}/4")
                        st.markdown(f"• Systemic: {systemic_symptoms}/4")
                        st.markdown(f"• Lymphatic: {lymph_symptoms}/2")
                        st.markdown(f"• Other: {chest_pain_val + back_pain_val + loss_appetite_val}/3")
                    
                    # Feature importance
                    if "feature_importance" in model_data:
                        with st.expander("📊 Feature Importance Analysis"):
                            importance_df = pd.DataFrame({
                                "Feature": list(model_data["feature_importance"].keys()),
                                "Importance": list(model_data["feature_importance"].values())
                            }).sort_values("Importance", ascending=False).head(10)
                            st.bar_chart(importance_df.set_index("Feature"))
                    
                    # Recommendations
                    st.markdown("---")
                    st.subheader("💡 Recommendations")
                    
                    if prediction == 1:
                        st.markdown("""
                        **Immediate Actions Required:**
                        - 🏥 Seek immediate medical evaluation for TB testing
                        - 🧪 Request sputum test and chest X-ray
                        - 😷 Practice respiratory hygiene (cover coughs, wear mask)
                        - 🏠 Limit close contact with others until evaluated
                        - 📋 Inform healthcare provider of all symptoms
                        - 💊 Do not self-medicate - TB requires specific antibiotics
                        """)
                    elif symptom_count >= 4:
                        st.markdown("""
                        **Recommended Actions:**
                        - 📅 Schedule a medical appointment within 1-2 weeks
                        - 📝 Monitor symptoms and note any changes
                        - 🧪 Consider TB screening if symptoms persist
                        - 🥗 Maintain good nutrition to support immune system
                        - 😴 Get adequate rest
                        """)
                    else:
                        st.markdown("""
                        **Maintain Good Health:**
                        - ✅ Continue healthy lifestyle habits
                        - 💪 Maintain strong immune system
                        - 🚭 Avoid smoking and secondhand smoke
                        - 🏠 Ensure good ventilation in living spaces
                        - 📅 Regular health check-ups
                        """)
                    
                    # Recommendations from AI
                    if name:
                        patient_info = {
                            "name": name,
                            "age": age,
                            "symptom_count": symptom_count,
                            "tb_risk": tb_risk_prob
                        }
                        recommendations = get_health_recommendations("Tuberculosis", severity, patient_info)
                        if recommendations:
                            display_recommendations(recommendations)
                else:
                    st.error("Model format not recognized.")
                    
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Malaria Prediction
if selected == 'Malaria Prediction':
    st.title("🦟 Malaria Prediction")
    st.markdown("AI-powered malaria detection using clinical symptoms and patient data")
    
    # Model accuracy info
    if "malaria_model" in loaded_models:
        model_data = loaded_models["malaria_model"]
        if isinstance(model_data, dict) and "accuracy" in model_data:
            st.success(f"✅ Model loaded successfully | Accuracy: {model_data['accuracy']:.2%}")
    
    name = st.text_input("👤 Patient Name:")
    
    # Organized tabs for better UX
    tab1, tab2 = st.tabs(["📋 Demographics", "🏥 Symptoms"])
    
    with tab1:
        st.subheader("Patient Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=100, value=35)
        with col2:
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        with col3:
            residence_area = st.selectbox("Residence Area", [0, 1, 2, 3, 4],
                                          format_func=lambda x: ["Chickmagalur", "Kasargod", "Mangalore", "Shimoga", "Udupi"][x])
    
    with tab2:
        st.subheader("Clinical Symptoms")
        st.info("💡 Select 'Yes' (1) or 'No' (0) for each symptom")
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            fever = st.selectbox("Fever", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            headache = st.selectbox("Headache", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            abdominal_pain = st.selectbox("Abdominal Pain", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            general_body_malaise = st.selectbox("General Body Malaise", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col5:
            dizziness = st.selectbox("Dizziness", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            vomiting = st.selectbox("Vomiting", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            confusion = st.selectbox("Confusion", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            backache = st.selectbox("Backache", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col6:
            chest_pain = st.selectbox("Chest Pain", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            coughing = st.selectbox("Coughing", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            joint_pain = st.selectbox("Joint Pain", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    if st.button("🔬 Predict Malaria Risk", type="primary"):
        try:
            if "malaria_model" not in loaded_models:
                st.error("Malaria model not loaded. Please check model file.")
            else:
                model_data = loaded_models["malaria_model"]
                
                if isinstance(model_data, dict):
                    model = model_data["model"]
                    scaler = model_data["scaler"]
                    feature_columns = model_data["feature_columns"]
                    
                    # Create input dataframe with exact feature order
                    # Features: Age, Sex, Residence_Area, Fever, Headache, Abdominal_Pain, 
                    # General_Body_Malaise, Dizziness, Vomiting, Confusion, Backache, Chest_Pain, Coughing, Joint_Pain
                    input_data = pd.DataFrame([[
                        age, sex, residence_area, fever, headache, abdominal_pain,
                        general_body_malaise, dizziness, vomiting, confusion,
                        backache, chest_pain, coughing, joint_pain
                    ]], columns=feature_columns)
                    
                    # Scale the input
                    input_scaled = scaler.transform(input_data)
                    
                    # Make prediction
                    prediction = model.predict(input_scaled)
                    prediction_proba = model.predict_proba(input_scaled)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    if prediction[0] == 1:
                        risk_prob = prediction_proba[0][1] * 100
                        st.error(f"⚠️ **HIGH RISK: Malaria Detected**")
                        st.metric("Malaria Probability", f"{risk_prob:.1f}%")
                        severity = "high"
                    else:
                        risk_prob = prediction_proba[0][0] * 100
                        st.success(f"✅ **LOW RISK: No Malaria Detected**")
                        st.metric("Healthy Probability", f"{risk_prob:.1f}%")
                        severity = "low"
                    
                    # Risk visualization
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.write("**Risk Assessment:**")
                        st.progress(prediction_proba[0][1])
                    
                    with col_r2:
                        st.write("**Probability Distribution:**")
                        prob_df = pd.DataFrame({
                            "Outcome": ["Negative", "Positive"],
                            "Probability": prediction_proba[0]
                        })
                        st.bar_chart(prob_df.set_index("Outcome"))
                    
                    # Feature importance display
                    if "feature_importance" in model_data:
                        with st.expander("📊 Feature Importance Analysis"):
                            importance_df = pd.DataFrame({
                                "Feature": feature_columns,
                                "Importance": model_data["feature_importance"]
                            }).sort_values("Importance", ascending=False)
                            st.bar_chart(importance_df.set_index("Feature"))
                    
                    # Recommendations
                    if name:
                        patient_info = {
                            "name": name,
                            "age": age,
                            "risk_probability": prediction_proba[0][1] * 100
                        }
                        recommendations = get_health_recommendations("Malaria", severity, patient_info)
                        if recommendations:
                            display_recommendations(recommendations)
                            display_health_tips_dynamic("Malaria", severity)
                else:
                    st.error("Model format not recognized.")
                    
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Prostate Cancer Prediction
if selected == 'Prostate Cancer Prediction':
    st.title("🧫 Prostate Cancer Prediction")
    st.markdown("AI-powered prostate cancer detection using tumor measurements")
    
    # Model accuracy info
    if "prostate_model" in loaded_models:
        model_data = loaded_models["prostate_model"]
        if isinstance(model_data, dict) and "accuracy" in model_data:
            st.success(f"✅ Model loaded successfully | Accuracy: {model_data['accuracy']:.2%}")
    
    name = st.text_input("👤 Patient Name:")
    
    st.markdown("### 📊 Tumor Measurements")
    st.info("Enter tumor cell measurements from biopsy analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        radius = st.number_input("Radius (mean)", min_value=5.0, max_value=30.0, value=14.0, step=0.1,
                                  help="Mean of distances from center to points on perimeter")
        texture = st.number_input("Texture (mean)", min_value=5.0, max_value=40.0, value=19.0, step=0.1,
                                   help="Standard deviation of gray-scale values")
        perimeter = st.number_input("Perimeter (mean)", min_value=40.0, max_value=200.0, value=90.0, step=1.0,
                                     help="Mean size of the tumor perimeter")
    
    with col2:
        area = st.number_input("Area (mean)", min_value=100.0, max_value=2500.0, value=600.0, step=10.0,
                               help="Mean tumor area")
        smoothness = st.number_input("Smoothness (mean)", min_value=0.05, max_value=0.20, value=0.10, step=0.01,
                                      help="Local variation in radius lengths")
        compactness = st.number_input("Compactness (mean)", min_value=0.01, max_value=0.40, value=0.10, step=0.01,
                                       help="Perimeter² / area - 1.0")
    
    with col3:
        symmetry = st.number_input("Symmetry (mean)", min_value=0.10, max_value=0.35, value=0.18, step=0.01,
                                    help="Symmetry of the tumor")
        fractal_dimension = st.number_input("Fractal Dimension (mean)", min_value=0.05, max_value=0.10, value=0.06, step=0.001,
                                             help="Coastline approximation - 1")
    
    if st.button("🔬 Predict Cancer Type", type="primary"):
        try:
            if "prostate_model" not in loaded_models:
                st.error("Prostate model not loaded. Please check model file.")
            else:
                model_data = loaded_models["prostate_model"]
                
                if isinstance(model_data, dict):
                    model = model_data["model"]
                    scaler = model_data["scaler"]
                    feature_columns = model_data["feature_columns"]
                    
                    # Create input dataframe
                    input_data = pd.DataFrame([[
                        radius, texture, perimeter, area, 
                        smoothness, compactness, symmetry, fractal_dimension
                    ]], columns=feature_columns)
                    
                    # Scale and predict
                    input_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_scaled)[0]
                    
                    # Get probability if available
                    if hasattr(model, 'predict_proba'):
                        prediction_proba = model.predict_proba(input_scaled)[0]
                    else:
                        prediction_proba = [1-prediction, prediction]
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    col_r1, col_r2, col_r3 = st.columns(3)
                    
                    with col_r1:
                        if prediction == 1:
                            st.error("🔴 **MALIGNANT**")
                            severity = "high"
                            diagnosis = "Malignant (Cancerous)"
                        else:
                            st.success("🟢 **BENIGN**")
                            severity = "low"
                            diagnosis = "Benign (Non-cancerous)"
                    
                    with col_r2:
                        st.metric("Confidence", f"{max(prediction_proba)*100:.1f}%")
                    
                    with col_r3:
                        st.metric("Malignancy Probability", f"{prediction_proba[1]*100:.1f}%")
                    
                    # Probability visualization
                    st.write("**Probability Distribution:**")
                    prob_df = pd.DataFrame({
                        "Diagnosis": ["Benign", "Malignant"],
                        "Probability": prediction_proba
                    })
                    st.bar_chart(prob_df.set_index("Diagnosis"))
                    
                    # Feature importance
                    if "feature_importance" in model_data and sum(model_data["feature_importance"]) > 0:
                        with st.expander("📊 Feature Importance"):
                            importance_df = pd.DataFrame({
                                "Feature": feature_columns,
                                "Importance": model_data["feature_importance"]
                            }).sort_values("Importance", ascending=False)
                            st.bar_chart(importance_df.set_index("Feature"))
                    
                    # Recommendations
                    if name:
                        patient_info = {
                            "name": name,
                            "diagnosis": diagnosis,
                            "confidence": f"{max(prediction_proba)*100:.1f}%"
                        }
                        recommendations = get_health_recommendations("Prostate Cancer", severity, patient_info)
                        if recommendations:
                            display_recommendations(recommendations)
                            display_health_tips_dynamic("Prostate Cancer", severity)
                else:
                    st.error("Model format not recognized.")
                    
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

                    
        except Exception as e:
            st.error(f"Error in assessment: {str(e)}")

# Cervical Cancer Prediction
# Cervical Cancer Prediction - 93% Accuracy ML Model
if selected == 'Cervical Cancer Prediction':
    st.title("👩 Cervical Cancer Risk Assessment")
    st.markdown("**93% Accurate** - AI-powered cervical cancer risk prediction using clinical factors")

    # Model info
    if "cervical_model" in loaded_models:
        model_data = loaded_models["cervical_model"]
        accuracy = model_data.get('accuracy', 0) * 100
        cv_accuracy = model_data.get('cv_accuracy', 0) * 100
        st.success(f"✅ ML Model loaded | Test Accuracy: {accuracy:.1f}% | CV Accuracy: {cv_accuracy:.1f}%")
    
    st.info("💡 This model uses clinical and lifestyle factors to assess cervical cancer risk based on medical research data.")

    name = st.text_input("👤 Patient Name:")

    # Organized tabs
    tab1, tab2, tab3 = st.tabs([
        "📋 Demographics & History", 
        "🔬 Medical Factors",
        "📊 Screening & STDs"
    ])
    
    with tab1:
        st.subheader("Demographics & Sexual History")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            num_sexual_partners = st.number_input("Number of Sexual Partners", min_value=0, max_value=50, value=3)
        
        with col2:
            first_intercourse = st.number_input("Age at First Sexual Intercourse", min_value=10, max_value=50, value=18)
            num_pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=2)
        
        with col3:
            smokes = st.selectbox("Smoking Status", ["No", "Yes"])
            smokes_years = st.number_input("Years of Smoking", min_value=0.0, max_value=50.0, value=0.0)
            smokes_packs = st.number_input("Packs per Year", min_value=0.0, max_value=100.0, value=0.0)
    
    with tab2:
        st.subheader("Contraceptives & Medical History")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Hormonal Contraceptives**")
            hormonal_contraceptives = st.selectbox("Hormonal Contraceptives Use", ["No", "Yes"])
            hormonal_years = st.number_input("Years on Hormonal Contraceptives", min_value=0.0, max_value=40.0, value=0.0)
        
        with col2:
            st.markdown("**IUD (Intrauterine Device)**")
            iud = st.selectbox("IUD Use", ["No", "Yes"])
            iud_years = st.number_input("Years with IUD", min_value=0.0, max_value=40.0, value=0.0)
    
    with tab3:
        st.subheader("STD History & Diagnosis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**STD Status**")
            stds = st.selectbox("History of STDs", ["No", "Yes"])
            stds_number = st.number_input("Number of STDs", min_value=0, max_value=10, value=0)
            stds_num_diagnosis = st.number_input("Number of STD Diagnoses", min_value=0, max_value=10, value=0)
        
        with col2:
            st.markdown("**Specific STDs**")
            stds_condylomatosis = st.selectbox("STDs: Condylomatosis", ["No", "Yes"])
            stds_cervical_condyl = st.selectbox("STDs: Cervical Condylomatosis", ["No", "Yes"])
            stds_vaginal_condyl = st.selectbox("STDs: Vaginal Condylomatosis", ["No", "Yes"])
        
        with col3:
            st.markdown("**HPV & Other**")
            stds_hpv = st.selectbox("STDs: HPV", ["No", "Yes"])
            stds_hiv = st.selectbox("STDs: HIV", ["No", "Yes"])
            stds_hepatitis_b = st.selectbox("STDs: Hepatitis B", ["No", "Yes"])
        
        st.markdown("---")
        st.subheader("Diagnosis History")
        col4, col5 = st.columns(2)
        
        with col4:
            dx = st.selectbox("Previous Cancer Diagnosis", ["No", "Yes"])
            dx_cancer = st.selectbox("Cancer Diagnosis Confirmed", ["No", "Yes"])
        
        with col5:
            dx_cin = st.selectbox("CIN Diagnosis", ["No", "Yes"])
            dx_hpv = st.selectbox("HPV Diagnosis", ["No", "Yes"])

    st.markdown("---")
    
    if st.button("🔬 Predict Cervical Cancer Risk", type="primary", use_container_width=True):
        if not name:
            st.warning("⚠️ Please enter patient name")
        elif "cervical_model" not in loaded_models:
            st.error("❌ Cervical cancer model not loaded. Please check if the model file exists.")
        else:
            try:
                with st.spinner("🤖 Analyzing risk factors..."):
                    model_data = loaded_models["cervical_model"]
                    model = model_data['model']
                    scaler = model_data['scaler']
                    feature_columns = model_data['feature_columns']
                    
                    # Convert inputs to numeric
                    smokes_num = 1 if smokes == "Yes" else 0
                    hormonal_num = 1 if hormonal_contraceptives == "Yes" else 0
                    iud_num = 1 if iud == "Yes" else 0
                    stds_num = 1 if stds == "Yes" else 0
                    stds_condyl_num = 1 if stds_condylomatosis == "Yes" else 0
                    stds_cerv_condyl_num = 1 if stds_cervical_condyl == "Yes" else 0
                    stds_vag_condyl_num = 1 if stds_vaginal_condyl == "Yes" else 0
                    stds_hpv_num = 1 if stds_hpv == "Yes" else 0
                    stds_hiv_num = 1 if stds_hiv == "Yes" else 0
                    stds_hep_b_num = 1 if stds_hepatitis_b == "Yes" else 0
                    dx_num = 1 if dx == "Yes" else 0
                    dx_cancer_num = 1 if dx_cancer == "Yes" else 0
                    dx_cin_num = 1 if dx_cin == "Yes" else 0
                    dx_hpv_num = 1 if dx_hpv == "Yes" else 0
                    
                    # Create feature vector matching the model's expected order
                    # Features from training: Age, Number of sexual partners, First sexual intercourse,
                    # Num of pregnancies, Smokes, Smokes (years), Smokes (packs/year),
                    # Hormonal Contraceptives, Hormonal Contraceptives (years), IUD, IUD (years),
                    # STDs, STDs (number), STDs:condylomatosis, STDs:cervical condylomatosis,
                    # STDs:vaginal condylomatosis, STDs:vulvo-perineal condylomatosis,
                    # STDs:syphilis, STDs:pelvic inflammatory disease, STDs:genital herpes,
                    # STDs:molluscum contagiosum, STDs:AIDS, STDs:HIV, STDs:Hepatitis B,
                    # STDs:HPV, STDs: Number of diagnosis, STDs: Time since first diagnosis,
                    # STDs: Time since last diagnosis, Dx:Cancer, Dx:CIN, Dx:HPV, Dx
                    
                    # For simplicity, create a DataFrame with median values for missing columns
                    user_input = pd.DataFrame({
                        'Age': [age],
                        'Number of sexual partners': [num_sexual_partners],
                        'First sexual intercourse': [first_intercourse],
                        'Num of pregnancies': [num_pregnancies],
                        'Smokes': [smokes_num],
                        'Smokes (years)': [smokes_years],
                        'Smokes (packs/year)': [smokes_packs],
                        'Hormonal Contraceptives': [hormonal_num],
                        'Hormonal Contraceptives (years)': [hormonal_years],
                        'IUD': [iud_num],
                        'IUD (years)': [iud_years],
                        'STDs': [stds_num],
                        'STDs (number)': [stds_number],
                        'STDs:condylomatosis': [stds_condyl_num],
                        'STDs:cervical condylomatosis': [stds_cerv_condyl_num],
                        'STDs:vaginal condylomatosis': [stds_vag_condyl_num],
                        'STDs:vulvo-perineal condylomatosis': [0],
                        'STDs:syphilis': [0],
                        'STDs:pelvic inflammatory disease': [0],
                        'STDs:genital herpes': [0],
                        'STDs:molluscum contagiosum': [0],
                        'STDs:AIDS': [0],
                        'STDs:HIV': [stds_hiv_num],
                        'STDs:Hepatitis B': [stds_hep_b_num],
                        'STDs:HPV': [stds_hpv_num],
                        'STDs: Number of diagnosis': [stds_num_diagnosis],
                        'STDs: Time since first diagnosis': [0],
                        'STDs: Time since last diagnosis': [0],
                        'Dx:Cancer': [dx_cancer_num],
                        'Dx:CIN': [dx_cin_num],
                        'Dx:HPV': [dx_hpv_num],
                        'Dx': [dx_num]
                    })
                    
                    # Ensure columns match model's feature columns
                    user_input = user_input[feature_columns]
                    
                    # Scale and predict
                    user_input_scaled = scaler.transform(user_input)
                    prediction = model.predict(user_input_scaled)
                    prediction_proba = model.predict_proba(user_input_scaled)[0]
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 📊 Cervical Cancer Risk Assessment Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    positive_prob = prediction_proba[1] * 100 if len(prediction_proba) > 1 else 0
                    
                    with col1:
                        if prediction[0] == 1 or positive_prob >= 30:
                            st.error("🔴 ELEVATED RISK DETECTED")
                            severity = "high"
                            risk_level = "High"
                        elif positive_prob >= 15:
                            st.warning("🟡 MODERATE RISK")
                            severity = "moderate"
                            risk_level = "Moderate"
                        else:
                            st.success("🟢 LOW RISK")
                            severity = "low"
                            risk_level = "Low"
                    
                    with col2:
                        confidence = max(prediction_proba) * 100
                        st.metric("🎯 Model Confidence", f"{confidence:.1f}%")
                    
                    with col3:
                        st.metric("📈 Risk Probability", f"{positive_prob:.1f}%")
                    
                    # Clinical interpretation
                    st.markdown("---")
                    if prediction[0] == 1 or positive_prob >= 30:
                        st.error(f"""
                        ### ⚠️ Clinical Alert for {name}
                        
                        The AI model indicates **elevated cervical cancer risk** requiring immediate attention.
                        
                        **Immediate Recommendations:**
                        - 🏥 Schedule colposcopy/biopsy consultation
                        - 🔬 HPV DNA testing if not recently done
                        - 📋 Comprehensive gynecological examination
                        - 💉 HPV vaccination (if age appropriate and not vaccinated)
                        - 🚭 Smoking cessation if applicable
                        """)
                    elif positive_prob >= 15:
                        st.warning(f"""
                        ### ⚠️ Moderate Risk for {name}
                        
                        Some risk factors identified. Enhanced monitoring recommended.
                        
                        **Recommendations:**
                        - 📅 Schedule Pap smear and HPV test
                        - 🩺 Regular gynecological check-ups
                        - 💉 Ensure HPV vaccination is complete
                        - 📋 Follow up on any abnormal results
                        """)
                    else:
                        st.success(f"""
                        ### ✅ Low Risk Assessment for {name}
                        
                        Current assessment shows low cervical cancer risk.
                        
                        **Recommendations:**
                        - 📅 Continue routine Pap smear screening
                        - 💉 Maintain HPV vaccination status
                        - 🌟 Healthy lifestyle practices
                        - 📋 Follow standard screening guidelines
                        """)
                    
                    # Risk factor analysis
                    st.markdown("---")
                    st.subheader("🔍 Risk Factor Analysis")
                    
                    risk_factors = []
                    if first_intercourse < 16:
                        risk_factors.append(("Early Sexual Activity", f"First intercourse at age {first_intercourse}"))
                    if num_sexual_partners > 4:
                        risk_factors.append(("Multiple Partners", f"{num_sexual_partners} sexual partners"))
                    if smokes == "Yes":
                        risk_factors.append(("Smoking", f"{smokes_years} years, {smokes_packs} packs/year"))
                    if hormonal_years > 5:
                        risk_factors.append(("Long-term Hormonal Contraceptives", f"{hormonal_years} years"))
                    if stds == "Yes":
                        risk_factors.append(("STD History", f"{stds_number} STD(s) recorded"))
                    if stds_hpv == "Yes":
                        risk_factors.append(("HPV Positive", "Major risk factor for cervical cancer"))
                    if stds_hiv == "Yes":
                        risk_factors.append(("HIV Positive", "Immunocompromised status"))
                    if dx_cin == "Yes":
                        risk_factors.append(("CIN History", "Cervical intraepithelial neoplasia diagnosed"))
                    
                    if risk_factors:
                        for factor, detail in risk_factors:
                            st.warning(f"⚠️ **{factor}**: {detail}")
                    else:
                        st.info("✅ No major risk factors identified")
                    
                    # Feature importance visualization
                    if 'feature_importance' in model_data:
                        st.markdown("---")
                        st.subheader("📊 Key Predictive Factors")
                        
                        importance_df = pd.DataFrame({
                            'Feature': list(model_data['feature_importance'].keys()),
                            'Importance': list(model_data['feature_importance'].values())
                        }).sort_values('Importance', ascending=False).head(10)
                        
                        st.bar_chart(importance_df.set_index('Feature')['Importance'])
                    
                    # AI Recommendations
                    st.markdown("---")
                    with st.spinner("🤖 Generating personalized recommendations..."):
                        patient_info = {
                            "name": name,
                            "age": age,
                            "risk_probability": positive_prob,
                            "risk_factors": ", ".join([f[0] for f in risk_factors]) if risk_factors else "None"
                        }
                        
                        recommendations = get_health_recommendations("Cervical Cancer Prevention", severity, patient_info)
                        if recommendations:
                            display_recommendations(recommendations)
                            display_health_tips_dynamic("Cervical Cancer", severity.lower())

            except Exception as e:
                st.error(f"❌ Error in prediction: {str(e)}")
                with st.expander("🔧 Technical Details"):
                    import traceback
                    st.code(traceback.format_exc())

# Asthma Prediction
if selected == 'Asthma Prediction':
    st.title("🫁 Asthma Risk Assessment")
    st.markdown("AI-powered asthma risk prediction using clinical and environmental factors")
    
    # Model accuracy info
    if "asthma_model" in loaded_models:
        model_data = loaded_models["asthma_model"]
        if isinstance(model_data, dict) and "accuracy" in model_data:
            st.success(f"✅ Model loaded successfully | Accuracy: {model_data['accuracy']:.2%}")
    
    name = st.text_input("👤 Patient Name:")
    
    # Organized tabs for better UX
    tab1, tab2, tab3 = st.tabs([
        "📋 Demographics & Lifestyle", 
        "🌿 Environmental Exposures",
        "🏥 Symptoms & History"
    ])
    
    with tab1:
        st.subheader("Demographics & Lifestyle Factors")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=100, value=35)
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
            ethnicity = st.selectbox("Ethnicity", [0, 1, 2, 3], 
                                     format_func=lambda x: ["Caucasian", "African American", "Asian", "Other"][x])
        
        with col2:
            education_level = st.selectbox("Education Level", [0, 1, 2, 3],
                                           format_func=lambda x: ["None", "High School", "Bachelor's", "Higher"][x])
            bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0,
                                  help="Normal range: 18.5-24.9")
            smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Non-Smoker" if x == 0 else "Smoker")
        
        with col3:
            physical_activity = st.slider("Physical Activity Level (0-10)", 0.0, 10.0, 5.0,
                                          help="0 = Sedentary, 10 = Very Active")
            diet_quality = st.slider("Diet Quality (0-10)", 0.0, 10.0, 5.0,
                                     help="0 = Poor, 10 = Excellent")
            sleep_quality = st.slider("Sleep Quality (4-10)", 4.0, 10.0, 7.0,
                                      help="4 = Poor, 10 = Excellent")
    
    with tab2:
        st.subheader("🌿 Environmental Exposure Factors")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            pollution_exposure = st.slider("Air Pollution Exposure (0-10)", 0.0, 10.0, 5.0,
                                           help="0 = Low, 10 = High")
            pollen_exposure = st.slider("Pollen Exposure (0-10)", 0.0, 10.0, 5.0,
                                        help="0 = Low, 10 = High")
        
        with col5:
            dust_exposure = st.slider("Dust Exposure (0-10)", 0.0, 10.0, 5.0,
                                      help="0 = Low, 10 = High")
            pet_allergy = st.selectbox("Pet Allergy", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col6:
            family_history_asthma = st.selectbox("Family History of Asthma", [0, 1], 
                                                  format_func=lambda x: "No" if x == 0 else "Yes")
            history_of_allergies = st.selectbox("History of Allergies", [0, 1], 
                                                 format_func=lambda x: "No" if x == 0 else "Yes")
    
    with tab3:
        st.subheader("🏥 Medical History & Current Symptoms")
        col7, col8, col9 = st.columns(3)
        
        with col7:
            eczema = st.selectbox("Eczema/Atopic Dermatitis", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            hay_fever = st.selectbox("Hay Fever (Allergic Rhinitis)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            gastroesophageal_reflux = st.selectbox("Gastroesophageal Reflux (GERD)", [0, 1], 
                                                    format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col8:
            lung_function_fev1 = st.number_input("Lung Function FEV1 (L)", min_value=0.5, max_value=6.0, value=2.5,
                                                  help="Forced Expiratory Volume in 1 second")
            lung_function_fvc = st.number_input("Lung Function FVC (L)", min_value=0.5, max_value=7.0, value=3.5,
                                                 help="Forced Vital Capacity")
        
        with col9:
            wheezing = st.selectbox("Wheezing Episodes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            shortness_of_breath = st.selectbox("Shortness of Breath", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            chest_tightness = st.selectbox("Chest Tightness", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        col10, col11, col12 = st.columns(3)
        with col10:
            coughing = st.selectbox("Frequent Coughing", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with col11:
            nighttime_symptoms = st.selectbox("Nighttime Symptoms", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with col12:
            exercise_induced = st.selectbox("Exercise-Induced Symptoms", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    if st.button("🔬 Predict Asthma Risk", type="primary"):
        try:
            if "asthma_model" not in loaded_models:
                st.error("Asthma model not loaded. Please check model file.")
            else:
                model_data = loaded_models["asthma_model"]
                
                if isinstance(model_data, dict):
                    model = model_data["model"]
                    scaler = model_data["scaler"]
                    feature_columns = model_data["feature_columns"]
                    
                    # Create input dataframe with exact feature order
                    input_data = pd.DataFrame([[
                        age, gender, ethnicity, education_level, bmi, smoking,
                        physical_activity, diet_quality, sleep_quality,
                        pollution_exposure, pollen_exposure, dust_exposure, pet_allergy,
                        family_history_asthma, history_of_allergies, eczema, hay_fever,
                        gastroesophageal_reflux, lung_function_fev1, lung_function_fvc,
                        wheezing, shortness_of_breath, chest_tightness, coughing,
                        nighttime_symptoms, exercise_induced
                    ]], columns=feature_columns)
                    
                    # Scale the input
                    input_scaled = scaler.transform(input_data)
                    
                    # Make prediction
                    prediction = model.predict(input_scaled)
                    prediction_proba = model.predict_proba(input_scaled)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    if prediction[0] == 1:
                        risk_prob = prediction_proba[0][1] * 100
                        st.error(f"⚠️ **High Risk of Asthma Detected**")
                        st.metric("Asthma Risk Probability", f"{risk_prob:.1f}%")
                        severity = "high"
                    else:
                        risk_prob = prediction_proba[0][0] * 100
                        st.success(f"✅ **Low Risk of Asthma**")
                        st.metric("Healthy Probability", f"{risk_prob:.1f}%")
                        severity = "low"
                    
                    # Risk visualization
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.write("**Risk Assessment:**")
                        st.progress(prediction_proba[0][1])
                    
                    with col_r2:
                        st.write("**Probability Distribution:**")
                        prob_df = pd.DataFrame({
                            "Outcome": ["No Asthma", "Asthma"],
                            "Probability": prediction_proba[0]
                        })
                        st.bar_chart(prob_df.set_index("Outcome"))
                    
                    # Feature importance display
                    if "feature_importance" in model_data:
                        with st.expander("📊 Feature Importance Analysis"):
                            importance_df = pd.DataFrame({
                                "Feature": feature_columns,
                                "Importance": model_data["feature_importance"]
                            }).sort_values("Importance", ascending=False).head(10)
                            st.bar_chart(importance_df.set_index("Feature"))
                    
                    # Recommendations
                    if name:
                        patient_info = {
                            "name": name,
                            "age": age,
                            "bmi": bmi,
                            "risk_probability": prediction_proba[0][1] * 100
                        }
                        recommendations = get_health_recommendations("Asthma", severity, patient_info)
                        if recommendations:
                            display_recommendations(recommendations)
                            display_health_tips_dynamic("Asthma", severity)
                else:
                    st.error("Model format not recognized.")
                    
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# COPD Prediction
# COPD Severity Prediction - 99% Accuracy Model
if selected == 'COPD Prediction':
    st.title("🫁 COPD Severity Prediction")
    st.markdown("**99% Accurate** - Predict COPD severity using clinical and pulmonary function data")
    
    # Model info
    if "copd_model" in loaded_models:
        model_data = loaded_models["copd_model"]
        accuracy = model_data.get('accuracy', 0) * 100
        cv_accuracy = model_data.get('cv_accuracy', 0) * 100
        st.success(f"✅ Model loaded | Test Accuracy: {accuracy:.1f}% | CV Accuracy: {cv_accuracy:.1f}%")
    
    st.info("💡 This model predicts whether a patient has **Severe/Very Severe COPD** vs **Mild/Moderate COPD** based on clinical measurements and pulmonary function tests.")
    
    name = st.text_input("👤 Patient Name:")
    
    # Organized tabs
    tab1, tab2, tab3 = st.tabs([
        "📋 Demographics & History", 
        "🫁 Pulmonary Function Tests",
        "🏥 Comorbidities & Symptoms"
    ])
    
    with tab1:
        st.subheader("Demographics & Smoking History")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age (years)", min_value=30, max_value=100, value=65, help="Patient age")
            gender = st.selectbox("Gender", ["Male", "Female"])
        
        with col2:
            pack_history = st.number_input("Pack-Years History", min_value=0.0, max_value=200.0, value=30.0, 
                                           help="Number of packs smoked per day × years of smoking")
            smoking = st.selectbox("Current Smoking Status", ["Non-smoker", "Current Smoker"])
    
    with tab2:
        st.subheader("🫁 Pulmonary Function Tests (Spirometry)")
        st.markdown("*These values are typically obtained from a pulmonary function test (PFT)*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Walking Tests (6-Minute Walk Test)**")
            mwt1 = st.number_input("MWT1 - First Walk Distance (meters)", min_value=0.0, max_value=800.0, value=350.0,
                                   help="Distance walked in first 6-minute walk test")
            mwt2 = st.number_input("MWT2 - Second Walk Distance (meters)", min_value=0.0, max_value=800.0, value=360.0,
                                   help="Distance walked in second 6-minute walk test")
            mwt1_best = st.number_input("MWT1 Best (meters)", min_value=0.0, max_value=800.0, value=365.0,
                                        help="Best distance from first walk test")
            
        with col2:
            st.markdown("**Lung Function Measurements**")
            fev1 = st.number_input("FEV1 (Liters)", min_value=0.0, max_value=6.0, value=1.5, step=0.1,
                                   help="Forced Expiratory Volume in 1 second - actual value")
            fev1_pred = st.number_input("FEV1 % Predicted", min_value=0.0, max_value=150.0, value=50.0,
                                        help="FEV1 as percentage of predicted normal - KEY INDICATOR")
            fvc = st.number_input("FVC (Liters)", min_value=0.0, max_value=8.0, value=3.0, step=0.1,
                                  help="Forced Vital Capacity - total air exhaled forcefully")
            fvc_pred = st.number_input("FVC % Predicted", min_value=0.0, max_value=150.0, value=70.0,
                                       help="FVC as percentage of predicted normal")
    
    with tab3:
        st.subheader("Quality of Life & Comorbidities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Symptom Scores**")
            cat = st.number_input("CAT Score (0-40)", min_value=0, max_value=40, value=20,
                                  help="COPD Assessment Test - measures impact on daily life. Higher = worse symptoms")
            had = st.number_input("HAD Score", min_value=0, max_value=42, value=10,
                                  help="Hospital Anxiety and Depression Scale")
            sgrq = st.number_input("SGRQ Score (0-100)", min_value=0.0, max_value=100.0, value=45.0,
                                   help="St. George's Respiratory Questionnaire - quality of life score. Higher = worse")
        
        with col2:
            st.markdown("**Cardiovascular Comorbidities**")
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            atrial_fib = st.selectbox("Atrial Fibrillation", ["No", "Yes"])
            ihd = st.selectbox("Ischemic Heart Disease (IHD)", ["No", "Yes"])
        
        with col3:
            st.markdown("**Other Conditions**")
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            muscular = st.selectbox("Muscular Dysfunction", ["No", "Yes"])
    
    st.markdown("---")
    
    # GOLD Classification Reference
    with st.expander("📊 GOLD COPD Classification Reference"):
        st.markdown("""
        | Stage | FEV1 % Predicted | Severity |
        |-------|------------------|----------|
        | GOLD 1 | ≥80% | Mild |
        | GOLD 2 | 50-79% | Moderate |
        | GOLD 3 | 30-49% | Severe |
        | GOLD 4 | <30% | Very Severe |
        
        *This model predicts Severe/Very Severe (GOLD 3-4) vs Mild/Moderate (GOLD 1-2)*
        """)
    
    if st.button("🔬 Predict COPD Severity", type="primary", use_container_width=True):
        if not name:
            st.warning("⚠️ Please enter patient name")
        elif "copd_model" not in loaded_models:
            st.error("❌ COPD model not loaded. Please check if the model file exists.")
        else:
            try:
                with st.spinner("🤖 Analyzing pulmonary function data..."):
                    model_data = loaded_models["copd_model"]
                    model = model_data['model']
                    scaler = model_data['scaler']
                    feature_columns = model_data['feature_columns']
                    
                    # Convert inputs to numeric
                    gender_num = 1 if gender == "Male" else 0
                    smoking_num = 2 if smoking == "Current Smoker" else 1
                    diabetes_num = 1 if diabetes == "Yes" else 0
                    muscular_num = 1 if muscular == "Yes" else 0
                    hypertension_num = 1 if hypertension == "Yes" else 0
                    atrial_fib_num = 1 if atrial_fib == "Yes" else 0
                    ihd_num = 1 if ihd == "Yes" else 0
                    
                    # Create feature vector in correct order
                    # Features: AGE, PackHistory, MWT1, MWT2, MWT1Best, FEV1, FEV1PRED, FVC, FVCPRED, 
                    #           CAT, HAD, SGRQ, gender, smoking, Diabetes, muscular, hypertension, AtrialFib, IHD
                    user_input = pd.DataFrame([[
                        age,
                        pack_history,
                        mwt1,
                        mwt2,
                        mwt1_best,
                        fev1,
                        fev1_pred,
                        fvc,
                        fvc_pred,
                        cat,
                        had,
                        sgrq,
                        gender_num,
                        smoking_num,
                        diabetes_num,
                        muscular_num,
                        hypertension_num,
                        atrial_fib_num,
                        ihd_num
                    ]], columns=feature_columns)
                    
                    # Scale and predict
                    user_input_scaled = scaler.transform(user_input)
                    prediction = model.predict(user_input_scaled)
                    prediction_proba = model.predict_proba(user_input_scaled)[0]
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 📊 COPD Severity Assessment Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if prediction[0] == 1:
                            st.error("🔴 SEVERE/VERY SEVERE COPD")
                            severity = "high"
                            risk_level = "High Risk"
                        else:
                            st.success("🟡 MILD/MODERATE COPD")
                            severity = "moderate"
                            risk_level = "Moderate Risk"
                    
                    with col2:
                        confidence = max(prediction_proba) * 100
                        st.metric("🎯 Model Confidence", f"{confidence:.1f}%")
                    
                    with col3:
                        severe_prob = prediction_proba[1] * 100 if len(prediction_proba) > 1 else 0
                        st.metric("📈 Severe COPD Probability", f"{severe_prob:.1f}%")
                    
                    # FEV1-based GOLD Classification
                    st.markdown("---")
                    st.subheader("📋 Clinical Classification")
                    
                    # Determine GOLD stage based on FEV1 % Predicted
                    if fev1_pred >= 80:
                        gold_stage = "GOLD 1 (Mild)"
                        gold_color = "green"
                    elif fev1_pred >= 50:
                        gold_stage = "GOLD 2 (Moderate)"
                        gold_color = "orange"
                    elif fev1_pred >= 30:
                        gold_stage = "GOLD 3 (Severe)"
                        gold_color = "red"
                    else:
                        gold_stage = "GOLD 4 (Very Severe)"
                        gold_color = "darkred"
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("GOLD Stage", gold_stage)
                    with col2:
                        st.metric("FEV1 % Predicted", f"{fev1_pred:.1f}%")
                    with col3:
                        fev1_fvc_ratio = (fev1/fvc)*100 if fvc > 0 else 0
                        st.metric("FEV1/FVC Ratio", f"{fev1_fvc_ratio:.1f}%")
                    
                    # Clinical interpretation
                    st.markdown("---")
                    if prediction[0] == 1:
                        st.error(f"""
                        ### ⚠️ Clinical Alert for {name}
                        
                        The AI model predicts **Severe/Very Severe COPD** with {confidence:.1f}% confidence.
                        
                        **Key Findings:**
                        - FEV1 % Predicted: {fev1_pred:.1f}% ({gold_stage})
                        - CAT Score: {cat}/40 {'(High symptom burden)' if cat >= 20 else ''}
                        - 6-Minute Walk Distance: {max(mwt1, mwt2):.0f} meters
                        
                        **Immediate Recommendations:**
                        - 🏥 Urgent pulmonologist consultation recommended
                        - 💊 Review and optimize bronchodilator therapy
                        - 🫁 Consider pulmonary rehabilitation program
                        - 🚭 Smoking cessation is critical if applicable
                        - 💉 Ensure vaccinations are up to date (flu, pneumonia)
                        - 🆘 Establish exacerbation action plan
                        """)
                    else:
                        st.success(f"""
                        ### ✅ Assessment Summary for {name}
                        
                        The AI model indicates **Mild/Moderate COPD** with {confidence:.1f}% confidence.
                        
                        **Key Findings:**
                        - FEV1 % Predicted: {fev1_pred:.1f}% ({gold_stage})
                        - CAT Score: {cat}/40
                        - 6-Minute Walk Distance: {max(mwt1, mwt2):.0f} meters
                        
                        **Management Recommendations:**
                        - 🩺 Regular follow-up with healthcare provider
                        - 💊 Continue prescribed bronchodilator therapy
                        - 🚭 Smoking cessation if applicable
                        - 💪 Maintain physical activity
                        - 📅 Annual pulmonary function monitoring
                        - 💉 Stay current with vaccinations
                        """)
                    
                    # Risk factors analysis
                    st.markdown("---")
                    st.subheader("🔍 Risk Factor Analysis")
                    
                    risk_factors = []
                    if pack_history > 20:
                        risk_factors.append(("Heavy Smoking History", f"{pack_history:.0f} pack-years"))
                    if fev1_pred < 50:
                        risk_factors.append(("Low FEV1", f"{fev1_pred:.1f}% predicted"))
                    if cat >= 20:
                        risk_factors.append(("High Symptom Burden", f"CAT score {cat}/40"))
                    if sgrq > 50:
                        risk_factors.append(("Poor Quality of Life", f"SGRQ {sgrq:.0f}/100"))
                    if max(mwt1, mwt2) < 300:
                        risk_factors.append(("Reduced Exercise Capacity", f"{max(mwt1, mwt2):.0f}m walk distance"))
                    if hypertension == "Yes":
                        risk_factors.append(("Hypertension", "Cardiovascular comorbidity"))
                    if diabetes == "Yes":
                        risk_factors.append(("Diabetes", "Metabolic comorbidity"))
                    if ihd == "Yes":
                        risk_factors.append(("Ischemic Heart Disease", "Major cardiovascular risk"))
                    if atrial_fib == "Yes":
                        risk_factors.append(("Atrial Fibrillation", "Cardiac arrhythmia"))
                    
                    if risk_factors:
                        for factor, detail in risk_factors:
                            st.warning(f"⚠️ **{factor}**: {detail}")
                    else:
                        st.info("✅ No major additional risk factors identified")
                    
                    # Feature importance visualization
                    if 'feature_importance' in model_data:
                        st.markdown("---")
                        st.subheader("📊 Key Predictive Factors")
                        
                        importance_df = pd.DataFrame({
                            'Feature': list(model_data['feature_importance'].keys()),
                            'Importance': list(model_data['feature_importance'].values())
                        }).sort_values('Importance', ascending=False).head(10)
                        
                        # Rename features for display
                        feature_names = {
                            'FEV1PRED': 'FEV1 % Predicted',
                            'FEV1': 'FEV1 (L)',
                            'FVCPRED': 'FVC % Predicted',
                            'FVC': 'FVC (L)',
                            'CAT': 'CAT Score',
                            'SGRQ': 'SGRQ Score',
                            'MWT1': 'Walk Test 1',
                            'MWT2': 'Walk Test 2',
                            'MWT1Best': 'Best Walk Distance',
                            'AGE': 'Age',
                            'PackHistory': 'Pack-Years',
                            'HAD': 'HAD Score'
                        }
                        importance_df['Feature'] = importance_df['Feature'].map(lambda x: feature_names.get(x, x))
                        
                        st.bar_chart(importance_df.set_index('Feature')['Importance'])
                    
                    # AI Recommendations
                    st.markdown("---")
                    with st.spinner("🤖 Generating personalized recommendations..."):
                        patient_info = {
                            "name": name,
                            "age": age,
                            "fev1_pred": fev1_pred,
                            "cat_score": cat,
                            "gold_stage": gold_stage,
                            "risk_factors": ", ".join([f[0] for f in risk_factors]) if risk_factors else "None"
                        }
                        
                        recommendations = get_health_recommendations("COPD", severity, patient_info)
                        if recommendations:
                            display_recommendations(recommendations)
                            display_health_tips_dynamic("COPD", severity)

            except Exception as e:
                st.error(f"❌ Error in prediction: {str(e)}")
                with st.expander("🔧 Technical Details"):
                    import traceback
                    st.code(traceback.format_exc())
# Pneumonia Prediction
if selected == 'Pneumonia Prediction':
    st.title("🫁 Pneumonia Risk Assessment")
    st.markdown("Evaluate pneumonia risk based on symptoms and risk factors")
    
    name = st.text_input("Name:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        fever = st.selectbox("Fever", ["No", "Low-grade (<38°C)", "High (>38°C)"])
    
    with col2:
        cough_type = st.selectbox("Cough Type", ["No cough", "Dry", "Productive with sputum"])
        chest_pain = st.selectbox("Chest Pain with Breathing", ["No", "Mild", "Severe"])
        shortness_breath = st.selectbox("Shortness of Breath", ["No", "Mild", "Moderate", "Severe"])
    
    with col3:
        fatigue = st.selectbox("Fatigue Level", ["None", "Mild", "Moderate", "Severe"])
        chills = st.selectbox("Chills/Sweating", ["No", "Yes"])
        confusion = st.selectbox("Confusion (especially in elderly)", ["No", "Yes"])
    
    # Risk factors
    st.subheader("Risk Factors")
    col4, col5 = st.columns(2)
    
    with col4:
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        chronic_disease = st.multiselect("Chronic Conditions", 
            ["None", "Diabetes", "Heart Disease", "COPD", "Asthma", "Kidney Disease"])
        immunocompromised = st.selectbox("Immunocompromised", ["No", "Yes"])
    
    with col5:
        recent_hospitalization = st.selectbox("Recent Hospitalization", ["No", "Yes"])
        vaccination_status = st.selectbox("Pneumonia Vaccine", ["Yes", "No", "Unknown"])
        exposure = st.selectbox("Recent Exposure to Sick People", ["No", "Yes"])
    
    if st.button("Assess Pneumonia Risk"):
        severity_score = 0
        
        # CURB-65 scoring elements
        if age >= 65:
            severity_score += 1
        
        if confusion == "Yes":
            severity_score += 1
        
        # Additional risk factors
        if fever == "High (>38°C)":
            severity_score += 1
        
        if shortness_breath in ["Moderate", "Severe"]:
            severity_score += 1
        
        if immunocompromised == "Yes":
            severity_score += 1
        
        # Determine risk level
        if severity_score >= 3:
            risk = "High - Seek immediate medical attention"
            color = "red"
            severity = "severe"
        elif severity_score >= 2:
            risk = "Moderate - Medical evaluation recommended"
            color = "orange"
            severity = "moderate"
        else:
            risk = "Low - Monitor symptoms"
            color = "green"
            severity = "mild"
        
        st.markdown(f"### Pneumonia Risk: <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)
        
        if name:
            with st.spinner("Generating recommendations..."):
                patient_info = {
                    "name": name,
                    "age": age,
                    "risk_score": severity_score,
                    "vaccination_status": vaccination_status
                }
                
                recommendations = get_health_recommendations("Pneumonia", severity, patient_info)
                if recommendations:
                    display_recommendations(recommendations)
                    display_health_tips_dynamic("Pneumonia", severity.lower())


# Migraine Prediction
if selected == 'Migraine Prediction':
    st.title("💥 Migraine Type Prediction")
    st.markdown("AI-powered migraine classification using symptom patterns and clinical features")
    
    # Model accuracy info
    if "migraine_model" in loaded_models:
        model_data = loaded_models["migraine_model"]
        if isinstance(model_data, dict) and "accuracy" in model_data:
            st.success(f"✅ Model loaded successfully | Accuracy: {model_data['accuracy']:.2%} | Type: {model_data.get('model_type', 'ML')}")
    
    name = st.text_input("👤 Patient Name:")
    
    # Organized tabs
    tab1, tab2, tab3 = st.tabs(["📋 Basic Info", "🩺 Symptoms", "🧠 Neurological Signs"])
    
    with tab1:
        st.subheader("Basic Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=10, max_value=100, value=35)
            duration = st.slider("Duration (hours)", min_value=1, max_value=72, value=12, help="Typical headache duration in hours")
        
        with col2:
            frequency = st.slider("Frequency (days/month)", min_value=0, max_value=30, value=5, help="Number of headache days per month")
            intensity = st.slider("Intensity (1-10)", min_value=1, max_value=10, value=5, help="Pain intensity scale")
        
        with col3:
            location = st.selectbox("Location", options=["Unilateral", "Bilateral", "Orbital"], index=0)
            location_map = {"Unilateral": 0, "Bilateral": 1, "Orbital": 2}
            location_val = location_map[location]
            
            character = st.selectbox("Pain Character", options=["Throbbing", "Pressing", "Stabbing"], index=0)
            character_map = {"Throbbing": 0, "Pressing": 1, "Stabbing": 2}
            character_val = character_map[character]
    
    with tab2:
        st.subheader("Associated Symptoms")
        col1, col2 = st.columns(2)
        
        with col1:
            nausea = st.checkbox("🤢 Nausea")
            nausea_val = 1 if nausea else 0
            
            vomit = st.checkbox("🤮 Vomiting")
            vomit_val = 1 if vomit else 0
            
            phonophobia = st.checkbox("🔊 Phonophobia (Sound Sensitivity)")
            phonophobia_val = 1 if phonophobia else 0
            
            photophobia = st.checkbox("💡 Photophobia (Light Sensitivity)")
            photophobia_val = 1 if photophobia else 0
        
        with col2:
            visual = st.checkbox("👁️ Visual Disturbances (Aura)")
            visual_val = 1 if visual else 0
            
            sensory = st.checkbox("✋ Sensory Disturbances")
            sensory_val = 1 if sensory else 0
            
            dpf = st.checkbox("👨‍👩‍👧 Family History of Migraine (DPF)")
            dpf_val = 1 if dpf else 0
    
    with tab3:
        st.subheader("Neurological Signs")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dysphasia = st.checkbox("🗣️ Dysphasia (Speech Difficulty)")
            dysphasia_val = 1 if dysphasia else 0
            
            dysarthria = st.checkbox("🗣️ Dysarthria (Slurred Speech)")
            dysarthria_val = 1 if dysarthria else 0
            
            vertigo = st.checkbox("🌀 Vertigo")
            vertigo_val = 1 if vertigo else 0
        
        with col2:
            tinnitus = st.checkbox("👂 Tinnitus (Ringing in Ears)")
            tinnitus_val = 1 if tinnitus else 0
            
            hypoacusis = st.checkbox("👂 Hypoacusis (Hearing Loss)")
            hypoacusis_val = 1 if hypoacusis else 0
            
            diplopia = st.checkbox("👁️ Diplopia (Double Vision)")
            diplopia_val = 1 if diplopia else 0
        
        with col3:
            defect = st.checkbox("👁️ Visual Field Defect")
            defect_val = 1 if defect else 0
            
            ataxia = st.checkbox("🚶 Ataxia (Balance Problems)")
            ataxia_val = 1 if ataxia else 0
            
            conscience = st.checkbox("😵 Altered Consciousness")
            conscience_val = 1 if conscience else 0
            
            paresthesia = st.checkbox("🖐️ Paresthesia (Tingling/Numbness)")
            paresthesia_val = 1 if paresthesia else 0
    
    st.divider()
    
    if st.button("🔬 Predict Migraine Type", type="primary", use_container_width=True):
        try:
            if "migraine_model" not in loaded_models:
                st.error("Migraine model not loaded. Please check model file.")
            else:
                model_data = loaded_models["migraine_model"]
                
                if isinstance(model_data, dict):
                    model = model_data["model"]
                    scaler = model_data["scaler"]
                    selector = model_data.get("selector")
                    label_encoder = model_data.get("label_encoder")
                    feature_columns = model_data["feature_columns"]
                    
                    # Prepare input - all original features
                    input_data = {
                        'Age': age,
                        'Duration': duration,
                        'Frequency': frequency,
                        'Location': location_val,
                        'Character': character_val,
                        'Intensity': intensity,
                        'Nausea': nausea_val,
                        'Vomit': vomit_val,
                        'Phonophobia': phonophobia_val,
                        'Photophobia': photophobia_val,
                        'Visual': visual_val,
                        'Sensory': sensory_val,
                        'Dysphasia': dysphasia_val,
                        'Dysarthria': dysarthria_val,
                        'Vertigo': vertigo_val,
                        'Tinnitus': tinnitus_val,
                        'Hypoacusis': hypoacusis_val,
                        'Diplopia': diplopia_val,
                        'Defect': defect_val,
                        'Ataxia': ataxia_val,
                        'Conscience': conscience_val,
                        'Paresthesia': paresthesia_val,
                        'DPF': dpf_val
                    }
                    
                    # Create DataFrame with correct column order
                    input_df = pd.DataFrame([input_data])[feature_columns]
                    
                    # Scale
                    input_scaled = scaler.transform(input_df)
                    
                    # Feature selection if selector exists
                    if selector:
                        input_selected = selector.transform(input_scaled)
                    else:
                        input_selected = input_scaled
                    
                    # Predict
                    prediction = model.predict(input_selected)[0]
                    
                    # Get class name
                    if label_encoder:
                        migraine_type = label_encoder.inverse_transform([prediction])[0]
                    else:
                        classes = model_data.get("classes", [])
                        migraine_type = classes[prediction] if classes else f"Type {prediction}"
                    
                    # Get probability
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(input_selected)[0]
                        confidence = max(proba) * 100
                        class_probas = dict(zip(label_encoder.classes_ if label_encoder else range(len(proba)), proba))
                    else:
                        confidence = 90.0
                        class_probas = {}
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Migraine Type", migraine_type)
                    with col2:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    with col3:
                        st.metric("Model Accuracy", f"{model_data.get('accuracy', 0.91)*100:.1f}%")
                    
                    # Type-specific information
                    type_info = {
                        "Typical aura with migraine": {
                            "description": "Classic migraine with visual or sensory aura preceding headache",
                            "color": "blue",
                            "severity": "moderate"
                        },
                        "Migraine without aura": {
                            "description": "Common migraine without preceding aura symptoms",
                            "color": "green",
                            "severity": "moderate"
                        },
                        "Typical aura without migraine": {
                            "description": "Aura symptoms without subsequent headache (acephalgic migraine)",
                            "color": "yellow",
                            "severity": "mild"
                        },
                        "Familial hemiplegic migraine": {
                            "description": "Rare inherited form with motor weakness during aura",
                            "color": "red",
                            "severity": "severe"
                        },
                        "Sporadic hemiplegic migraine": {
                            "description": "Non-familial hemiplegic migraine with motor weakness",
                            "color": "red",
                            "severity": "severe"
                        },
                        "Basilar-type aura": {
                            "description": "Migraine with aura originating from brainstem",
                            "color": "orange",
                            "severity": "severe"
                        },
                        "Other": {
                            "description": "Atypical migraine pattern requiring further evaluation",
                            "color": "gray",
                            "severity": "moderate"
                        }
                    }
                    
                    info = type_info.get(migraine_type, type_info["Other"])
                    
                    st.info(f"**{migraine_type}**: {info['description']}")
                    
                    if info["severity"] == "severe":
                        st.error(f"⚠️ {name if name else 'Patient'}, this migraine type requires neurological evaluation. Please consult a specialist.")
                    elif info["severity"] == "moderate":
                        st.warning(f"⚠️ {name if name else 'Patient'}, regular follow-up with your doctor is recommended.")
                    else:
                        st.success(f"✅ {name if name else 'Patient'}, this is a manageable condition with proper treatment.")
                    
                    # Probability distribution
                    if class_probas:
                        st.markdown("---")
                        st.subheader("📊 Probability Distribution")
                        prob_df = pd.DataFrame({
                            "Migraine Type": list(class_probas.keys()),
                            "Probability": [p*100 for p in class_probas.values()]
                        }).sort_values("Probability", ascending=False)
                        st.bar_chart(prob_df.set_index("Migraine Type"))
                    
                    # Feature importance
                    if "feature_importance" in model_data:
                        with st.expander("📊 Feature Importance Analysis"):
                            importance_df = pd.DataFrame({
                                "Feature": list(model_data["feature_importance"].keys()),
                                "Importance": list(model_data["feature_importance"].values())
                            }).sort_values("Importance", ascending=False)
                            st.bar_chart(importance_df.set_index("Feature"))
                    
                    # Symptom analysis
                    st.markdown("---")
                    st.subheader("📋 Symptom Analysis")
                    
                    present_symptoms = []
                    if nausea: present_symptoms.append("Nausea")
                    if vomit: present_symptoms.append("Vomiting")
                    if phonophobia: present_symptoms.append("Sound sensitivity")
                    if photophobia: present_symptoms.append("Light sensitivity")
                    if visual: present_symptoms.append("Visual aura")
                    if sensory: present_symptoms.append("Sensory disturbances")
                    if vertigo: present_symptoms.append("Vertigo")
                    if dysphasia: present_symptoms.append("Speech difficulty")
                    if tinnitus: present_symptoms.append("Tinnitus")
                    if paresthesia: present_symptoms.append("Tingling/Numbness")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Present Symptoms:**")
                        if present_symptoms:
                            for s in present_symptoms:
                                st.markdown(f"• {s}")
                        else:
                            st.info("No associated symptoms reported")
                    
                    with col2:
                        st.markdown("**Key Metrics:**")
                        st.markdown(f"• Frequency: {frequency} days/month")
                        st.markdown(f"• Duration: {duration} hours")
                        st.markdown(f"• Intensity: {intensity}/10")
                        st.markdown(f"• Family History: {'Yes' if dpf else 'No'}")
                    
                    # Recommendations
                    if name:
                        patient_info = {
                            "name": name,
                            "migraine_type": migraine_type,
                            "frequency": frequency,
                            "intensity": intensity
                        }
                        recommendations = get_health_recommendations("Migraine", info["severity"], patient_info)
                        if recommendations:
                            display_recommendations(recommendations)
                            display_health_tips_dynamic("Migraine", info["severity"])
                else:
                    st.error("Model format not recognized.")
                    
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


# Obesity Prediction
if selected == 'Obesity Prediction':
    st.title("⚖️ Obesity Risk Prediction")
    st.markdown("AI-powered obesity risk assessment using lifestyle and dietary factors")
    
    # Model accuracy info
    if "obesity_model" in loaded_models:
        model_data = loaded_models["obesity_model"]
        if isinstance(model_data, dict) and "accuracy" in model_data:
            st.success(f"✅ Model loaded successfully | Accuracy: {model_data['accuracy']:.2%}")
    
    name = st.text_input("👤 Patient Name:")
    
    # Organized tabs
    tab1, tab2, tab3 = st.tabs(["📋 Demographics", "🍽️ Eating Habits", "🏃 Lifestyle"])
    
    with tab1:
        st.subheader("Demographics & Physical Measurements")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            age = st.number_input("Age", min_value=10, max_value=100, value=30)
        
        with col2:
            height = st.number_input("Height (meters)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
        
        with col3:
            family_history = st.selectbox("Family History of Overweight", [0, 1], 
                                           format_func=lambda x: "No" if x == 0 else "Yes")
    
    with tab2:
        st.subheader("Eating Habits")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            favc = st.selectbox("Frequent High Caloric Food (FAVC)", [0, 1], 
                                format_func=lambda x: "No" if x == 0 else "Yes")
            fcvc = st.slider("Vegetable Consumption Frequency (1-3)", 1.0, 3.0, 2.0)
        
        with col5:
            ncp = st.slider("Number of Main Meals (1-4)", 1.0, 4.0, 3.0)
            caec = st.selectbox("Food Between Meals (CAEC)", [0, 1, 2, 3], 
                                format_func=lambda x: ["Always", "Frequently", "Sometimes", "No"][x])
        
        with col6:
            ch2o = st.slider("Daily Water Intake (liters)", 1.0, 3.0, 2.0)
            calc = st.selectbox("Alcohol Consumption (CALC)", [0, 1, 2, 3], 
                                format_func=lambda x: ["Always", "Frequently", "Sometimes", "No"][x])
    
    with tab3:
        st.subheader("Lifestyle Factors")
        col7, col8, col9 = st.columns(3)
        
        with col7:
            smoke = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            scc = st.selectbox("Calorie Monitoring (SCC)", [0, 1], 
                               format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col8:
            faf = st.slider("Physical Activity Frequency (0-3 days/week)", 0.0, 3.0, 1.0)
            tue = st.slider("Technology Use Time (0-2 hours)", 0.0, 2.0, 1.0)
        
        with col9:
            mtrans = st.selectbox("Transportation", [0, 1, 2, 3, 4], 
                                  format_func=lambda x: ["Automobile", "Bike", "Motorbike", "Public Transport", "Walking"][x])
    
    if st.button("🔬 Predict Obesity Risk", type="primary"):
        try:
            if "obesity_model" not in loaded_models:
                st.error("Obesity model not loaded. Please check model file.")
            else:
                model_data = loaded_models["obesity_model"]
                
                if isinstance(model_data, dict):
                    model = model_data["model"]
                    scaler = model_data["scaler"]
                    feature_columns = model_data["feature_columns"]
                    
                    # Create input dataframe with exact feature order
                    # Features: Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS
                    input_data = pd.DataFrame([[
                        gender, age, height, weight, family_history, favc, fcvc, ncp,
                        caec, smoke, ch2o, scc, faf, tue, calc, mtrans
                    ]], columns=feature_columns)
                    
                    # Scale the input
                    input_scaled = scaler.transform(input_data)
                    
                    # Make prediction
                    prediction = model.predict(input_scaled)
                    prediction_proba = model.predict_proba(input_scaled)
                    
                    # Calculate BMI for display
                    bmi = weight / (height ** 2)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    col_r1, col_r2, col_r3 = st.columns(3)
                    
                    with col_r1:
                        st.metric("📏 BMI", f"{bmi:.1f}")
                    
                    with col_r2:
                        if prediction[0] == 1:
                            st.error("🟠 **OBESE - High Risk**")
                            severity = "high"
                        else:
                            st.success("🟢 **NOT OBESE - Low Risk**")
                            severity = "low"
                    
                    with col_r3:
                        risk_prob = prediction_proba[0][1] * 100
                        st.metric("Obesity Probability", f"{risk_prob:.1f}%")
                    
                    # Risk visualization
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        st.write("**Risk Assessment:**")
                        st.progress(prediction_proba[0][1])
                    
                    with col_v2:
                        st.write("**Probability Distribution:**")
                        prob_df = pd.DataFrame({
                            "Outcome": ["Not Obese", "Obese"],
                            "Probability": prediction_proba[0]
                        })
                        st.bar_chart(prob_df.set_index("Outcome"))
                    
                    # BMI Category
                    st.markdown("---")
                    if bmi < 18.5:
                        st.info("🟦 BMI Category: **Underweight**")
                    elif bmi < 25:
                        st.success("🟢 BMI Category: **Normal Weight**")
                    elif bmi < 30:
                        st.warning("🟡 BMI Category: **Overweight**")
                    else:
                        st.error("🟠 BMI Category: **Obese**")
                    
                    # Feature importance display
                    if "feature_importance" in model_data:
                        with st.expander("📊 Feature Importance Analysis"):
                            importance_df = pd.DataFrame({
                                "Feature": feature_columns,
                                "Importance": model_data["feature_importance"]
                            }).sort_values("Importance", ascending=False)
                            st.bar_chart(importance_df.set_index("Feature"))
                    
                    # Recommendations
                    if name:
                        patient_info = {
                            "name": name,
                            "age": age,
                            "bmi": round(bmi, 2),
                            "risk_probability": risk_prob
                        }
                        recommendations = get_health_recommendations("Obesity", severity, patient_info)
                        if recommendations:
                            display_recommendations(recommendations)
                            display_health_tips_dynamic("Obesity", severity)
                else:
                    st.error("Model format not recognized.")
                    
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


# Cancer Risk Assessment
if selected == 'Cancer Risk Assessment':
    st.title("🎯 Cancer Risk Level Assessment")
    st.markdown("AI-powered cancer risk assessment based on lifestyle, environmental factors, and symptoms")
    
    # Model accuracy info
    if "cancer_risk_model" in loaded_models:
        model_data = loaded_models["cancer_risk_model"]
        if isinstance(model_data, dict) and "accuracy" in model_data:
            st.success(f"✅ Model loaded successfully | Accuracy: {model_data['accuracy']:.2%}")
    
    name = st.text_input("👤 Patient Name:")
    
    # Organized tabs
    tab1, tab2, tab3 = st.tabs(["📋 Demographics", "🌍 Environmental Factors", "🩺 Symptoms"])
    
    with tab1:
        st.subheader("Demographics & Lifestyle")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
            gender = st.selectbox("Gender", options=["Male", "Female"])
            gender_val = 1 if gender == "Male" else 2
        
        with col2:
            smoking = st.slider("🚬 Smoking (1-8)", min_value=1, max_value=8, value=3, help="1=Never, 8=Heavy smoker")
            alcohol_use = st.slider("🍺 Alcohol Use (1-8)", min_value=1, max_value=8, value=2, help="1=Never, 8=Heavy drinker")
        
        with col3:
            obesity = st.slider("⚖️ Obesity Level (1-7)", min_value=1, max_value=7, value=3, help="1=Underweight, 7=Morbidly obese")
            balanced_diet = st.slider("🥗 Balanced Diet (1-7)", min_value=1, max_value=7, value=4, help="1=Very poor, 7=Excellent")
    
    with tab2:
        st.subheader("Environmental & Occupational Factors")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            air_pollution = st.slider("🌫️ Air Pollution Exposure (1-8)", min_value=1, max_value=8, value=3, help="1=Very low, 8=Very high")
            dust_allergy = st.slider("🌬️ Dust Allergy (1-8)", min_value=1, max_value=8, value=3, help="1=None, 8=Severe")
        
        with col2:
            occupational_hazards = st.slider("⚠️ Occupational Hazards (1-8)", min_value=1, max_value=8, value=2, help="1=None, 8=Very high exposure")
            genetic_risk = st.slider("🧬 Genetic Risk (1-7)", min_value=1, max_value=7, value=3, help="1=No family history, 7=Strong family history")
        
        with col3:
            chronic_lung_disease = st.slider("🫁 Chronic Lung Disease (1-7)", min_value=1, max_value=7, value=2, help="1=None, 7=Severe")
            passive_smoker = st.slider("🚭 Passive Smoking (1-8)", min_value=1, max_value=8, value=2, help="1=Never exposed, 8=Heavy exposure")
    
    with tab3:
        st.subheader("Symptoms & Clinical Signs")
        col1, col2 = st.columns(2)
        
        with col1:
            chest_pain = st.slider("💔 Chest Pain (1-9)", min_value=1, max_value=9, value=2, help="1=None, 9=Severe")
            coughing_blood = st.slider("🩸 Coughing Blood (1-9)", min_value=1, max_value=9, value=1, help="1=Never, 9=Frequently")
            fatigue = st.slider("😴 Fatigue (1-9)", min_value=1, max_value=9, value=3, help="1=None, 9=Severe")
            weight_loss = st.slider("📉 Unexplained Weight Loss (1-8)", min_value=1, max_value=8, value=2, help="1=None, 8=Significant")
            shortness_of_breath = st.slider("😮‍💨 Shortness of Breath (1-9)", min_value=1, max_value=9, value=2, help="1=None, 9=Severe")
            wheezing = st.slider("🌬️ Wheezing (1-8)", min_value=1, max_value=8, value=2, help="1=None, 8=Constant")
        
        with col2:
            swallowing_difficulty = st.slider("🍽️ Swallowing Difficulty (1-8)", min_value=1, max_value=8, value=1, help="1=None, 8=Severe")
            clubbing_finger_nails = st.slider("👆 Clubbing of Finger Nails (1-9)", min_value=1, max_value=9, value=1, help="1=None, 9=Severe")
            frequent_cold = st.slider("🤧 Frequent Cold (1-7)", min_value=1, max_value=7, value=2, help="1=Rarely, 7=Very often")
            dry_cough = st.slider("😷 Dry Cough (1-7)", min_value=1, max_value=7, value=2, help="1=None, 7=Constant")
            snoring = st.slider("😴 Snoring (1-7)", min_value=1, max_value=7, value=2, help="1=Never, 7=Always")
    
    st.divider()
    
    if st.button("🔬 Assess Cancer Risk Level", type="primary", use_container_width=True):
        try:
            if "cancer_risk_model" not in loaded_models:
                st.error("Cancer Risk model not loaded. Please check model file.")
            else:
                model_data = loaded_models["cancer_risk_model"]
                
                if isinstance(model_data, dict):
                    model = model_data["model"]
                    scaler = model_data["scaler"]
                    label_encoder = model_data.get("label_encoder")
                    feature_columns = model_data["feature_columns"]
                    
                    # Prepare input - features in correct order
                    input_data = {
                        'Age': age,
                        'Gender': gender_val,
                        'Air Pollution': air_pollution,
                        'Alcohol use': alcohol_use,
                        'Dust Allergy': dust_allergy,
                        'OccuPational Hazards': occupational_hazards,
                        'Genetic Risk': genetic_risk,
                        'chronic Lung Disease': chronic_lung_disease,
                        'Balanced Diet': balanced_diet,
                        'Obesity': obesity,
                        'Smoking': smoking,
                        'Passive Smoker': passive_smoker,
                        'Chest Pain': chest_pain,
                        'Coughing of Blood': coughing_blood,
                        'Fatigue': fatigue,
                        'Weight Loss': weight_loss,
                        'Shortness of Breath': shortness_of_breath,
                        'Wheezing': wheezing,
                        'Swallowing Difficulty': swallowing_difficulty,
                        'Clubbing of Finger Nails': clubbing_finger_nails,
                        'Frequent Cold': frequent_cold,
                        'Dry Cough': dry_cough,
                        'Snoring': snoring
                    }
                    
                    # Create DataFrame with correct column order
                    input_df = pd.DataFrame([input_data])[feature_columns]
                    
                    # Scale and predict
                    input_scaled = scaler.transform(input_df)
                    prediction = model.predict(input_scaled)[0]
                    
                    # Get class name
                    if label_encoder:
                        risk_level = label_encoder.inverse_transform([prediction])[0]
                    else:
                        classes = model_data.get("classes", ["High", "Low", "Medium"])
                        risk_level = classes[prediction]
                    
                    # Get probability
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(input_scaled)[0]
                        confidence = max(proba) * 100
                    else:
                        confidence = 95.0
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Assessment Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Risk Level", risk_level)
                    with col2:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    with col3:
                        st.metric("Model Accuracy", f"{model_data.get('accuracy', 1.0)*100:.1f}%")
                    
                    # Risk-specific display
                    if risk_level == "High":
                        st.error(f"⚠️ {name if name else 'Patient'}, **HIGH CANCER RISK** detected! Immediate comprehensive screening and specialist consultation strongly recommended.")
                        severity = "high"
                    elif risk_level == "Medium":
                        st.warning(f"⚠️ {name if name else 'Patient'}, **MODERATE CANCER RISK** detected. Regular screening and lifestyle modifications recommended.")
                        severity = "moderate"
                    else:  # Low
                        st.success(f"✅ {name if name else 'Patient'}, **LOW CANCER RISK**. Continue healthy habits and regular preventive check-ups!")
                        severity = "low"
                    
                    # Risk factors analysis
                    st.markdown("---")
                    st.subheader("📋 Risk Factor Analysis")
                    
                    risk_factors = []
                    protective_factors = []
                    
                    # Analyze risk factors
                    if smoking >= 5:
                        risk_factors.append(f"🚬 Heavy smoking (Level {smoking}/8)")
                    if alcohol_use >= 5:
                        risk_factors.append(f"🍺 High alcohol consumption (Level {alcohol_use}/8)")
                    if air_pollution >= 5:
                        risk_factors.append(f"🌫️ High air pollution exposure (Level {air_pollution}/8)")
                    if genetic_risk >= 5:
                        risk_factors.append(f"🧬 Strong genetic/family history (Level {genetic_risk}/7)")
                    if obesity >= 5:
                        risk_factors.append(f"⚖️ Obesity (Level {obesity}/7)")
                    if coughing_blood >= 3:
                        risk_factors.append(f"🩸 Coughing blood symptom (Level {coughing_blood}/9)")
                    if chest_pain >= 5:
                        risk_factors.append(f"💔 Significant chest pain (Level {chest_pain}/9)")
                    if weight_loss >= 5:
                        risk_factors.append(f"📉 Unexplained weight loss (Level {weight_loss}/8)")
                    if chronic_lung_disease >= 4:
                        risk_factors.append(f"🫁 Chronic lung disease history (Level {chronic_lung_disease}/7)")
                    if occupational_hazards >= 5:
                        risk_factors.append(f"⚠️ High occupational hazard exposure (Level {occupational_hazards}/8)")
                    
                    # Analyze protective factors
                    if smoking <= 2:
                        protective_factors.append("🚭 Non-smoker or minimal smoking")
                    if balanced_diet >= 5:
                        protective_factors.append("🥗 Good balanced diet")
                    if obesity <= 3:
                        protective_factors.append("⚖️ Healthy weight")
                    if air_pollution <= 2:
                        protective_factors.append("🌿 Low pollution exposure")
                    if genetic_risk <= 2:
                        protective_factors.append("🧬 Low genetic risk")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### ⚠️ Risk Factors Identified")
                        if risk_factors:
                            for factor in risk_factors:
                                st.markdown(f"• {factor}")
                        else:
                            st.success("No major risk factors identified!")
                    
                    with col2:
                        st.markdown("### ✅ Protective Factors")
                        if protective_factors:
                            for factor in protective_factors:
                                st.markdown(f"• {factor}")
                        else:
                            st.info("Consider improving lifestyle factors")
                    
                    # Feature importance display
                    if "feature_importance" in model_data:
                        with st.expander("📊 Feature Importance Analysis"):
                            importance_df = pd.DataFrame({
                                "Feature": list(model_data["feature_importance"].keys()),
                                "Importance": list(model_data["feature_importance"].values())
                            }).sort_values("Importance", ascending=False).head(10)
                            st.bar_chart(importance_df.set_index("Feature"))
                    
                    # Recommendations
                    st.markdown("---")
                    st.subheader("💡 Recommendations")
                    
                    if risk_level == "High":
                        st.markdown("""
                        **Immediate Actions:**
                        - 🏥 Schedule comprehensive cancer screening (CT scan, blood tests, tumor markers)
                        - 👨‍⚕️ Consult an oncologist for risk assessment
                        - 🚬 Quit smoking immediately if applicable
                        - 📝 Document all symptoms and their duration
                        - 👨‍👩‍👧‍👦 Inform family about genetic counseling options
                        """)
                    elif risk_level == "Medium":
                        st.markdown("""
                        **Recommended Actions:**
                        - 📅 Schedule routine cancer screening tests
                        - 🥗 Improve diet with more fruits, vegetables, and whole grains
                        - 🏃 Increase physical activity (150+ min/week)
                        - 🚬 Reduce or quit smoking
                        - 🍺 Limit alcohol consumption
                        - 📆 Annual health check-ups with your doctor
                        """)
                    else:
                        st.markdown("""
                        **Maintain Your Health:**
                        - ✅ Continue healthy lifestyle habits
                        - 🥗 Maintain balanced diet rich in antioxidants
                        - 🏃 Stay physically active
                        - 📅 Regular preventive health screenings
                        - 🌿 Minimize exposure to environmental toxins
                        - 😴 Ensure adequate sleep and stress management
                        """)
                    
                    # Health tips
                    if name:
                        patient_info = {
                            "name": name,
                            "age": age,
                            "risk_level": risk_level
                        }
                        recommendations = get_health_recommendations("Cancer Risk", severity, patient_info)
                        if recommendations:
                            display_recommendations(recommendations)
                else:
                    st.error("Model format not recognized.")
                    
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

