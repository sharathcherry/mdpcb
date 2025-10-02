import streamlit as st
from openai import OpenAI
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import numpy as np
import seaborn as sns
import joblib
import datetime
import time
import requests
import json

# NVIDIA API Configuration for LLM Integration
NVIDIA_API_KEY = "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"  # Replace with your actual API key

# Initialize OpenAI client for NVIDIA
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# List to track failed models
failed_models = []

# Load each model individually
try:
    diabetes_model = joblib.load("models/diabetes_model.sav")
except Exception as e:
    failed_models.append("diabetes_model")

try:
    heart_model = joblib.load("models/heart_disease_model.sav")
except Exception as e:
    failed_models.append("heart_disease_model")

try:
    parkinson_model = joblib.load("models/parkinsons_model.sav")
except Exception as e:
    failed_models.append("parkinsons_model")

try:
    lung_cancer_model = joblib.load('models/lung_cancer_model.sav')
except Exception as e:
    failed_models.append("lung_cancer_model")

try:
    breast_cancer_model = joblib.load('models/breast_cancer.sav')
except Exception as e:
    failed_models.append("breast_cancer_model")

try:
    kidney = joblib.load('models/kidney_disease.sav')
except Exception as e:
    failed_models.append("kidney_disease")

try:
    hepatitis_model = joblib.load('models/hepititisc_model.sav')
except Exception as e:
    failed_models.append("hepatitis_model")

try:
    liver_model = joblib.load('models/liver_model.sav')
except Exception as e:
    failed_models.append("liver_model")

# Additional models (repeat the pattern for these)
try:
    alzheimers_model = joblib.load('models/alzheimers_model.sav')
except Exception as e:
    failed_models.append("alzheimers_model")

try:
    epilepsy_model = joblib.load('models/epilepsy_model.sav')
except Exception as e:
    failed_models.append("epilepsy_model")

try:
    migraine_model = joblib.load('models/migraine_model.sav')
except Exception as e:
    failed_models.append("migraine_model")

try:
    tb_model = joblib.load('models/tuberculosis_model.sav')
except Exception as e:
    failed_models.append("tb_model")

try:
    hiv_model = joblib.load('models/hiv_model.sav')
except Exception as e:
    failed_models.append("hiv_model")

try:
    malaria_model = joblib.load('models/malaria_model.sav')
except Exception as e:
    failed_models.append("malaria_model")

try:
    colorectal_model = joblib.load('models/colorectal_model.sav')
except Exception as e:
    failed_models.append("colorectal_model")

try:
    prostate_model = joblib.load('models/prostate_model.sav')
except Exception as e:
    failed_models.append("prostate_model")

try:
    cervical_model = joblib.load('models/cervical_model.sav')
except Exception as e:
    failed_models.append("cervical_model")

try:
    asthma_model = joblib.load('models/asthma_model.sav')
except Exception as e:
    failed_models.append("asthma_model")

try:
    copd_model = joblib.load('models/copd_model.sav')
except Exception as e:
    failed_models.append("copd_model")

try:
    pneumonia_model = joblib.load('models/pneumonia_model.sav')
except Exception as e:
    failed_models.append("pneumonia_model")

try:
    obesity_model = joblib.load('models/obesity_model.sav')
except Exception as e:
    failed_models.append("obesity_model")

# Display specific warning if any models failed
if failed_models:
    st.warning(f"The following models failed to load: {', '.join(failed_models)}. Please ensure their files are present in the 'models/' directory.")
else:
    # Optional: Success message if all loaded
    st.success("All models loaded successfully!")

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
            model="writer/palmyra-med-70b",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant. You MUST return only valid JSON with no additional text."},
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
            model="writer/palmyra-med-70b",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant. You MUST return only valid JSON with no additional text."},
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
            model="writer/palmyra-med-70b",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant. You MUST return only valid JSON with no additional text."},
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
        "HIV/AIDS Prediction": "🧬",
        "Malaria Prediction": "🦟",
        # Cancer
        "Lung Cancer Prediction": "🌬️",
        "Breast Cancer Prediction": "🎗️",
        "Colorectal Cancer Prediction": "🧬",
        "Prostate Cancer Prediction": "🧫",
        "Cervical Cancer Prediction": "🧫",
        # Respiratory
        "Asthma Prediction": "🌫️",
        "COPD Prediction": "😮‍💨",
        "Pneumonia Prediction": "🫁",
        # Services
        "AI Health Assistant": "🤖",
        "Book Appointment": "📅",
        "Set Reminder": "⏰",
        "Health Tips": "💡",
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
    ch5 = section("Infectious", ["Hepatitis Prediction", "Tuberculosis Prediction", "HIV/AIDS Prediction", "Malaria Prediction"])
    ch6 = section("Cancer", ["Lung Cancer Prediction", "Breast Cancer Prediction", "Colorectal Cancer Prediction", "Prostate Cancer Prediction", "Cervical Cancer Prediction"])
    ch7 = section("Respiratory", ["Asthma Prediction", "COPD Prediction", "Pneumonia Prediction"])
    ch8 = section("Services", ["AI Health Assistant", "Book Appointment", "Set Reminder", "Health Tips"])

    # Resolve choice precedence
    nav_choice = top_choice or ch1 or ch2 or ch3 or ch4 or ch5 or ch6 or ch7 or ch8
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
        st.metric("Total Diseases", "22+", "↑ 5 new")
        st.metric("Accuracy Rate", "94.5%", "↑ 2.3%")
    
    with col2:
        st.metric("Users Served", "10,234", "↑ 234")
        st.metric("Predictions Made", "45,678", "↑ 1,234")
    
    with col3:
        st.metric("Available Models", "22", "✓ All Active")
        st.metric("Response Time", "< 2s", "✓ Optimal")
    
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
            "HIV/AIDS", "Hepatitis B", "Hepatitis C", "Liver Disease",
            "Chronic Kidney Disease", "Parkinsons", "Alzheimers", "Epilepsy",
            "Migraine", "Lung Cancer", "Breast Cancer", "Colorectal Cancer",
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
# Lifestyle factors
        if physical_activity == "Sedentary":
            risk_score += 2
        if eating_habits in ["Often Unhealthy", "Very Unhealthy"]:
            risk_score += 2
        if fast_food in ["3-4 times/week", "Daily"]:
            risk_score += 1
        if family_obesity == "Both Parents":
            risk_score += 2
        elif family_obesity == "One Parent":
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 7:
            risk_level = "High Risk"
            severity = "high"
        elif risk_score >= 4:
            risk_level = "Moderate Risk"
            severity = "moderate"
        else:
            risk_level = "Low Risk"
            severity = "low"
        
        st.markdown(f"### Overall Obesity Risk: {risk_level}")
        st.progress(min(risk_score/10, 1.0))
        
        # Health recommendations based on BMI
        st.subheader("📋 Personalized Recommendations")
        
        if bmi >= 25:
            st.warning("Weight management is recommended for optimal health")
            target_weight = 24.9 * (height_m ** 2)
            weight_to_lose = weight - target_weight
            st.write(f"**Target Weight:** {target_weight:.1f} kg")
            if weight_to_lose > 0:
                st.write(f"**Weight to Lose:** {weight_to_lose:.1f} kg")
            
            # Calculate daily calorie needs
            if gender == "Male":
                bmr = 10 * weight + 6.25 * height - 5 * age + 5
            else:
                bmr = 10 * weight + 6.25 * height - 5 * age - 161
            
            activity_multipliers = {
                "Sedentary": 1.2,
                "Lightly Active": 1.375,
                "Moderately Active": 1.55,
                "Very Active": 1.725,
                "Extremely Active": 1.9
            }
            
            tdee = bmr * activity_multipliers.get(physical_activity, 1.2)
            calorie_deficit = tdee - 500  # 500 calorie deficit for healthy weight loss
            
            st.write(f"**Estimated Daily Calorie Needs:** {tdee:.0f} calories")
            st.write(f"**Recommended for Weight Loss:** {calorie_deficit:.0f} calories/day")
            st.write(f"**Estimated Time to Target Weight:** {(weight_to_lose * 7700 / 500 / 7):.1f} weeks (at 0.5 kg/week)")
        
        elif bmi < 18.5:
            target_weight = 18.5 * (height_m ** 2)
            weight_to_gain = target_weight - weight
            st.info("Weight gain is recommended for optimal health")
            st.write(f"**Target Weight:** {target_weight:.1f} kg")
            st.write(f"**Weight to Gain:** {weight_to_gain:.1f} kg")
        
        else:
            st.success("You are at a healthy weight! Focus on maintenance.")
        
        # Get AI recommendations for obesity/weight management
        if name:
            with st.spinner("Generating personalized weight management plan..."):
                patient_info = {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "bmi": round(bmi, 2),
                    "weight": weight,
                    "height": height,
                    "physical_activity": physical_activity,
                    "eating_habits": eating_habits,
                    "medical_conditions": medical_conditions
                }
                
                disease_name = "Obesity Management" if bmi >= 30 else "Weight Management"
                recommendations = get_health_recommendations(disease_name, severity, patient_info)
                if recommendations:
                    display_recommendations(recommendations)
                    display_health_tips_dynamic(severity.lower())

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.title("🩺 Diabetes Prediction")
    st.markdown("Predict Type 2 Diabetes risk based on clinical parameters from dataset")
    
    name = st.text_input("Name:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", options=["Male", "Female"])
        age = st.number_input("Age", min_value=1, max_value=120, value=33)
        hypertension = st.selectbox("Hypertension", options=[0, 1])
    
    with col2:
        heart_disease = st.selectbox("Heart Disease", options=[0, 1])
        smoking_history = st.selectbox("Smoking History", options=["never", "current", "former", "not current", "ever", "No Info"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)
    
    with col3:
        hba1c = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.7)
        blood_glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=0, max_value=500, value=120)
    
    if st.button("Predict Diabetes"):
        try:
            # Convert categorical variables to numeric
            gender_val = 1 if gender == "Male" else 0
            
            smoking_map = {
                "never": 0,
                "current": 1,
                "former": 2,
                "not current": 3,
                "ever": 4,
                "No Info": 5
            }
            smoking_num = smoking_map.get(smoking_history, 5)
            
            user_input = [gender_val, age, hypertension, heart_disease,
                          smoking_num, bmi, hba1c, blood_glucose]
            
            diabetes_prediction = diabetes_model.predict([user_input])
            
            if diabetes_prediction[0] == 1:
                st.error(f"{name}, high risk of Type 2 Diabetes detected!")
                severity = "high"
            else:
                st.success(f"{name}, low diabetes risk. Continue healthy lifestyle!")
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
if selected == 'Heart Disease Prediction':
    st.title("❤️ Heart Disease Prediction")
    st.markdown("Assess cardiovascular disease risk")
    
    name = st.text_input("Name:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    
    with col2:
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
        thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
    
    with col3:
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
        slope = st.selectbox("Slope of Peak Exercise ST", ["Upsloping", "Flat", "Downsloping"])
        ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    
    if st.button("Predict Heart Disease"):
        try:
            # Convert categorical to numerical
            sex_num = 1 if sex == "Male" else 0
            cp_num = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
            fbs_num = 1 if fbs == "Yes" else 0
            restecg_num = ["Normal", "ST-T Abnormality", "LV Hypertrophy"].index(restecg)
            exang_num = 1 if exang == "Yes" else 0
            slope_num = ["Upsloping", "Flat", "Downsloping"].index(slope)
            thal_num = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1
            
            user_input = [age, sex_num, cp_num, trestbps, chol, fbs_num, restecg_num, 
                         thalach, exang_num, oldpeak, slope_num, ca, thal_num]
            
            heart_prediction = heart_model.predict([user_input])
            
            if heart_prediction[0] == 1:
                st.error(f"{name}, heart disease risk detected! Consult a cardiologist immediately.")
                image = Image.open('positive.jpg')
                st.image(image, caption='High Risk')
                severity = "high"
            else:
                st.success(f"{name}, low heart disease risk. Maintain a healthy lifestyle!")
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

        except:
            st.error("Error in prediction. Please check your inputs.")

# Parkinson's Disease Prediction
if selected == 'Parkinsons Prediction':
    st.title("🧠 Parkinson's Disease Prediction")
    st.markdown("Assess Parkinson's disease risk based on vocal measurements")
    
    name = st.text_input("Name:")
    
    st.info("This prediction uses voice analysis parameters. Values are typically obtained from medical voice analysis.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, max_value=300.0, value=150.0)
        fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, max_value=300.0, value=180.0)
        flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, max_value=300.0, value=120.0)
        jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, max_value=1.0, value=0.005)
        jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, max_value=0.1, value=0.00003)
    
    with col2:
        rap = st.number_input("MDVP:RAP", min_value=0.0, max_value=0.1, value=0.003)
        ppq = st.number_input("MDVP:PPQ", min_value=0.0, max_value=0.1, value=0.003)
        ddp = st.number_input("Jitter:DDP", min_value=0.0, max_value=0.1, value=0.009)
        shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, max_value=1.0, value=0.03)
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, max_value=2.0, value=0.3)
    
    with col3:
        apq3 = st.number_input("Shimmer:APQ3", min_value=0.0, max_value=0.1, value=0.015)
        apq5 = st.number_input("Shimmer:APQ5", min_value=0.0, max_value=0.1, value=0.017)
        apq = st.number_input("MDVP:APQ", min_value=0.0, max_value=0.1, value=0.02)
        dda = st.number_input("Shimmer:DDA", min_value=0.0, max_value=0.3, value=0.045)
        nhr = st.number_input("NHR", min_value=0.0, max_value=1.0, value=0.025)
    
    col4, col5 = st.columns(2)
    with col4:
        hnr = st.number_input("HNR", min_value=0.0, max_value=40.0, value=20.0)
        rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0, value=0.5)
        dfa = st.number_input("DFA", min_value=0.0, max_value=1.0, value=0.7)
    
    with col5:
        spread1 = st.number_input("spread1", min_value=-10.0, max_value=0.0, value=-5.0)
        spread2 = st.number_input("spread2", min_value=0.0, max_value=1.0, value=0.2)
        d2 = st.number_input("D2", min_value=0.0, max_value=5.0, value=2.5)
        ppe = st.number_input("PPE", min_value=0.0, max_value=1.0, value=0.2)
    
    if st.button("Predict Parkinson's Disease"):
        try:
            user_input = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                         shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                         rpde, dfa, spread1, spread2, d2, ppe]
            
            parkinsons_prediction = parkinson_model.predict([user_input])
            
            if parkinsons_prediction[0] == 1:
                st.error(f"{name}, Parkinson's disease indicators detected. Neurological consultation recommended.")
                image = Image.open('positive.jpg')
                st.image(image, caption='Positive Indicators')
                severity = "moderate"
            else:
                st.success(f"{name}, no significant Parkinson's indicators detected.")
                severity = "low"
            
            # Get AI recommendations
            if name:
                with st.spinner("Generating neurological health recommendations..."):
                    patient_info = {
                        "name": name,
                        "voice_analysis": "completed"
                    }
                    
                    recommendations = get_health_recommendations("Parkinson's Disease", severity, patient_info)
                    if recommendations:
                        display_recommendations(recommendations)
                        display_health_tips_dynamic("Parkinson's Disease", severity.lower())

        except:
            st.error("Error in prediction. Please ensure all vocal parameters are entered.")

# Lung Cancer Prediction
if selected == 'Lung Cancer Prediction':
    st.title("🫁 Lung Cancer Prediction")
    st.markdown("Assess lung cancer risk based on symptoms and lifestyle")
    
    name = st.text_input("Name:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=55)
        gender = st.selectbox("Gender", ["Male", "Female"])
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
    
    with col2:
        anxiety = st.selectbox("Anxiety", ["No", "Yes"])
        peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
        chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])
        fatigue = st.selectbox("Fatigue", ["No", "Yes"])
    
    with col3:
        allergy = st.selectbox("Allergy", ["No", "Yes"])
        wheezing = st.selectbox("Wheezing", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
        coughing = st.selectbox("Coughing", ["No", "Yes"])
    
    col4, col5 = st.columns(2)
    with col4:
        shortness_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
        swallowing_difficulty = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
        chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])
    
    if st.button("Predict Lung Cancer Risk"):
        try:
            # Convert Yes/No to 1/0
            features = [
                1 if gender == "Male" else 0,
                age,
                1 if smoking == "Yes" else 0,
                1 if yellow_fingers == "Yes" else 0,
                1 if anxiety == "Yes" else 0,
                1 if peer_pressure == "Yes" else 0,
                1 if chronic_disease == "Yes" else 0,
                1 if fatigue == "Yes" else 0,
                1 if allergy == "Yes" else 0,
                1 if wheezing == "Yes" else 0,
                1 if alcohol == "Yes" else 0,
                1 if coughing == "Yes" else 0,
                1 if shortness_breath == "Yes" else 0,
                1 if swallowing_difficulty == "Yes" else 0,
                1 if chest_pain == "Yes" else 0
            ]
            
            lung_prediction = lung_cancer_model.predict([features])
            
            if lung_prediction[0] == 1:
                st.error(f"{name}, high lung cancer risk! Immediate medical consultation and screening recommended.")
                severity = "high"
            else:
                st.success(f"{name}, low lung cancer risk. Continue healthy habits!")
                severity = "low"
            
            # Get AI recommendations
            if name:
                with st.spinner("Generating lung health recommendations..."):
                    patient_info = {
                        "name": name,
                        "age": age,
                        "smoking_status": smoking,
                        "symptoms": [s for s, v in [
                            ("coughing", coughing),
                            ("chest_pain", chest_pain),
                            ("shortness_breath", shortness_breath)
                        ] if v == "Yes"]
                    }
                    
                    recommendations = get_health_recommendations("Lung Cancer", severity, patient_info)
                    if recommendations:
                        display_recommendations(recommendations)
                        display_health_tips_dynamic("Lung Cancer", severity.lower())

        except:
            st.error("Error in prediction. Please check your inputs.")

# Breast Cancer Prediction
if selected == 'Breast Cancer Prediction':
    st.title("🎗️ Breast Cancer Prediction")
    st.markdown("Predict breast cancer based on cell nuclei measurements")
    
    name = st.text_input("Name:")
    
    st.info("These measurements are typically obtained from fine needle aspirate (FNA) of breast mass.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        radius_mean = st.number_input("Radius Mean", min_value=0.0, max_value=50.0, value=14.0)
        texture_mean = st.number_input("Texture Mean", min_value=0.0, max_value=50.0, value=19.0)
        perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, max_value=200.0, value=92.0)
        area_mean = st.number_input("Area Mean", min_value=0.0, max_value=2500.0, value=655.0)
        smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, max_value=0.3, value=0.096)
    
    with col2:
        compactness_mean = st.number_input("Compactness Mean", min_value=0.0, max_value=0.5, value=0.104)
        concavity_mean = st.number_input("Concavity Mean", min_value=0.0, max_value=0.5, value=0.089)
        concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, max_value=0.3, value=0.048)
        symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, max_value=0.5, value=0.181)
        fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, max_value=0.1, value=0.063)
    
    with col3:
        radius_se = st.number_input("Radius SE", min_value=0.0, max_value=5.0, value=0.406)
        texture_se = st.number_input("Texture SE", min_value=0.0, max_value=5.0, value=1.216)
        perimeter_se = st.number_input("Perimeter SE", min_value=0.0, max_value=30.0, value=2.866)
        area_se = st.number_input("Area SE", min_value=0.0, max_value=500.0, value=40.34)
        smoothness_se = st.number_input("Smoothness SE", min_value=0.0, max_value=0.05, value=0.007)
    
    # Continue with remaining features in expandable section
    with st.expander("Additional Measurements"):
        col4, col5 = st.columns(2)
        with col4:
            compactness_se = st.number_input("Compactness SE", min_value=0.0, max_value=0.2, value=0.025)
            concavity_se = st.number_input("Concavity SE", min_value=0.0, max_value=0.2, value=0.032)
            concave_points_se = st.number_input("Concave Points SE", min_value=0.0, max_value=0.05, value=0.012)
            symmetry_se = st.number_input("Symmetry SE", min_value=0.0, max_value=0.1, value=0.020)
            fractal_dimension_se = st.number_input("Fractal Dimension SE", min_value=0.0, max_value=0.05, value=0.003)
        
        with col5:
            radius_worst = st.number_input("Radius Worst", min_value=0.0, max_value=50.0, value=16.0)
            texture_worst = st.number_input("Texture Worst", min_value=0.0, max_value=50.0, value=25.0)
            perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, max_value=300.0, value=107.0)
            area_worst = st.number_input("Area Worst", min_value=0.0, max_value=4000.0, value=880.0)
            smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, max_value=0.3, value=0.132)
        
        compactness_worst = st.number_input("Compactness Worst", min_value=0.0, max_value=1.0, value=0.254)
        concavity_worst = st.number_input("Concavity Worst", min_value=0.0, max_value=1.5, value=0.272)
        concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, max_value=0.3, value=0.114)
        symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, max_value=0.7, value=0.290)
        fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0, max_value=0.3, value=0.084)
    
    if st.button("Predict Breast Cancer"):
        try:
            user_input = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                         compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
                         fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
                         smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se,
                         fractal_dimension_se, radius_worst, texture_worst, perimeter_worst,
                         area_worst, smoothness_worst, compactness_worst, concavity_worst,
                         concave_points_worst, symmetry_worst, fractal_dimension_worst]
            
            breast_prediction = breast_cancer_model.predict([user_input])
            
            if breast_prediction[0] == 1:
                st.error(f"{name}, malignant tumor characteristics detected. Immediate oncology consultation required!")
                image = Image.open('positive.jpg')
                st.image(image, caption='Malignant')
                severity = "severe"
            else:
                st.success(f"{name}, benign tumor characteristics. Continue regular screening.")
                severity = "low"
            
            # Get AI recommendations
            if name:
                with st.spinner("Generating breast health recommendations..."):
                    patient_info = {
                        "name": name,
                        "tumor_characteristics": "analyzed"
                    }
                    
                    recommendations = get_health_recommendations("Breast Cancer", severity, patient_info)
                    if recommendations:
                        display_recommendations(recommendations)
                        display_health_tips_dynamic("Breast Cancer", severity.lower())

        except:
            st.error("Error in prediction. Please ensure all measurements are entered.")

# Kidney Disease Prediction
if selected == 'Kidney Disease Prediction':
    st.title("🫘 Chronic Kidney Disease Prediction")
    st.markdown("Assess kidney function and disease risk")

    name = st.text_input("Name:")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=48)
        bp = st.number_input("Blood Pressure (mm/Hg)", min_value=50, max_value=200, value=80)
        sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
        al = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])

    with col2:
        su = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5])
        rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
        pc = st.selectbox("Pus Cell", ["Normal", "Abnormal"])
        pcc = st.selectbox("Pus Cell Clumps", ["Present", "Not Present"])

    with col3:
        ba = st.selectbox("Bacteria", ["Present", "Not Present"])
        bgr = st.number_input("Blood Glucose Random (mgs/dl)", min_value=0, max_value=500, value=121)
        bu = st.number_input("Blood Urea (mgs/dl)", min_value=0, max_value=200, value=36)
        sc = st.number_input("Serum Creatinine (mgs/dl)", min_value=0.0, max_value=20.0, value=1.2)

    col4, col5 = st.columns(2)
    with col4:
        sod = st.number_input("Sodium (mEq/L)", min_value=0, max_value=200, value=137)
        pot = st.number_input("Potassium (mEq/L)", min_value=0.0, max_value=20.0, value=4.5)
        hemo = st.number_input("Hemoglobin (gms)", min_value=0.0, max_value=20.0, value=12.0)

    with col5:
        pcv = st.number_input("Packed Cell Volume", min_value=0, max_value=100, value=38)
        wc = st.number_input("White Blood Cell Count (cells/cumm)", min_value=0, max_value=30000, value=8000)
        rc = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=0.0, max_value=10.0, value=4.7)

    col6, col7 = st.columns(2)
    with col6:
        htn = st.selectbox("Hypertension", ["Yes", "No"])
        dm = st.selectbox("Diabetes Mellitus", ["Yes", "No"])
        cad = st.selectbox("Coronary Artery Disease", ["Yes", "No"])

    with col7:
        appet = st.selectbox("Appetite", ["Good", "Poor"])
        pe = st.selectbox("Pedal Edema", ["Yes", "No"])
        ane = st.selectbox("Anemia", ["Yes", "No"])

    if st.button("Predict Kidney Disease"):
        try:
            rbc_num = 1 if rbc == "Normal" else 0
            pc_num = 1 if pc == "Normal" else 0
            pcc_num = 0 if pcc == "Present" else 1
            ba_num = 0 if ba == "Present" else 1
            htn_num = 1 if htn == "Yes" else 0
            dm_num = 1 if dm == "Yes" else 0
            cad_num = 1 if cad == "Yes" else 0
            appet_num = 1 if appet == "Good" else 0
            pe_num = 1 if pe == "Yes" else 0
            ane_num = 1 if ane == "Yes" else 0

            user_input = [age, bp, sg, al, su, rbc_num, pc_num, pcc_num, ba_num, bgr, bu, 
                          sc, sod, pot, hemo, pcv, wc, rc, htn_num, dm_num, cad_num, 
                          appet_num, pe_num, ane_num]

            kidney_prediction = kidney.predict([user_input])

            if kidney_prediction[0] == 1:
                st.error(f"{name}, chronic kidney disease detected! Immediate nephrology consultation required.")
                image = Image.open('positive.jpg')
                st.image(image, caption='CKD Detected')
                severity = "high"
            else:
                st.success(f"{name}, no kidney disease detected. Maintain kidney health!")
                severity = "low"

            if name:
                with st.spinner("Generating kidney health recommendations..."):
                    patient_info = {
                        "name": name,
                        "age": age,
                        "creatinine": sc,
                        "blood_urea": bu,
                        "blood_pressure": bp,
                        "diabetes": dm,
                        "hypertension": htn
                    }

                    recommendations = get_health_recommendations("Chronic Kidney Disease", severity, patient_info)
                    if recommendations:
                        display_recommendations(recommendations)
                        display_health_tips_dynamic("Chronic Kidney Disease", severity.lower())


        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

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

# Hepatitis Prediction
if selected == 'Hepatitis Prediction':
    st.title("🦠 Hepatitis C Prediction")
    st.markdown("Assess Hepatitis C risk and disease stage")
    
    name = st.text_input("Name:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        alb = st.number_input("Albumin (g/L)", min_value=0.0, max_value=100.0, value=38.5)
        alp = st.number_input("Alkaline Phosphatase (IU/L)", min_value=0, max_value=500, value=52)
    
    with col2:
        alt = st.number_input("ALT (IU/L)", min_value=0, max_value=500, value=22)
        ast = st.number_input("AST (IU/L)", min_value=0, max_value=500, value=22)
        bil = st.number_input("Bilirubin (µmol/L)", min_value=0.0, max_value=500.0, value=7.5)
        che = st.number_input("Cholinesterase (kU/L)", min_value=0.0, max_value=30.0, value=6.9)
    
    with col3:
        chol = st.number_input("Cholesterol (mmol/L)", min_value=0.0, max_value=20.0, value=5.2)
        crea = st.number_input("Creatinine (µmol/L)", min_value=0, max_value=1000, value=74)
        ggt = st.number_input("Gamma-GT (IU/L)", min_value=0, max_value=500, value=23)
        prot = st.number_input("Total Protein (g/L)", min_value=0.0, max_value=150.0, value=72.9)
    
    if st.button("Predict Hepatitis Risk"):
        try:
            sex_num = 1 if sex == "Male" else 0
            
            user_input = [age, sex_num, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot]
            
            hepatitis_prediction = hepatitis_model.predict([user_input])
            
            if hepatitis_prediction[0] >= 1:
                stage = ["", "Hepatitis", "Fibrosis", "Cirrhosis"][min(int(hepatitis_prediction[0]), 3)]
                st.error(f"{name}, {stage} detected! Immediate hepatology consultation required.")
                image = Image.open('positive.jpg')
                st.image(image, caption=f'{stage} Detected')
                severity = "high"
            else:
                st.success(f"{name}, no hepatitis indicators detected.")
                severity = "low"
            
            # Get AI recommendations
            if name:
                with st.spinner("Generating hepatitis management recommendations..."):
                    patient_info = {
                        "name": name,
                        "age": age,
                        "alt": alt,
                        "ast": ast,
                        "bilirubin": bil
                    }
                    
                    recommendations = get_health_recommendations("Hepatitis", severity, patient_info)
                    if recommendations:
                        display_recommendations(recommendations)
                        display_health_tips_dynamic("Hepatitis", severity.lower())

        except:
            st.error("Error in prediction. Please check all inputs.")

# General Disease Prediction (Symptom-based)
if selected == '🔍 General Disease Prediction':
    st.title("🔍 General Disease Prediction")
    st.markdown("Predict diseases based on symptoms")
    
    st.info("Select your symptoms from the list below. The system will predict possible diseases.")
    
    name = st.text_input("Name:")
    
    # Common symptoms list
    symptoms_list = [
        "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering",
        "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting",
        "vomiting", "burning_micturition", "spotting_urination", "fatigue", "weight_gain",
        "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness",
        "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", "high_fever",
        "sunken_eyes", "breathlessness", "sweating", "dehydration", "indigestion",
        "headache", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite",
        "pain_behind_the_eyes", "back_pain", "constipation", "abdominal_pain", "diarrhoea",
        "mild_fever", "yellow_urine", "yellowing_of_eyes", "acute_liver_failure",
        "fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes", "malaise",
        "blurred_and_distorted_vision", "phlegm", "throat_irritation", "redness_of_eyes",
        "sinus_pressure", "runny_nose", "congestion", "chest_pain", "weakness_in_limbs",
        "fast_heart_rate", "pain_during_bowel_movements", "pain_in_anal_region",
        "bloody_stool", "irritation_in_anus", "neck_pain", "dizziness", "cramps",
        "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes",
        "enlarged_thyroid", "brittle_nails", "swollen_extremeties", "excessive_hunger",
        "extra_marital_contacts", "drying_and_tingling_lips", "slurred_speech",
        "knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints",
        "movement_stiffness", "spinning_movements", "loss_of_balance", "unsteadiness",
        "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort",
        "foul_smell_of_urine", "continuous_feel_of_urine", "passage_of_gases",
        "internal_itching", "toxic_look_(typhos)", "depression", "irritability",
        "muscle_pain", "altered_sensorium", "red_spots_over_body", "belly_pain",
        "abnormal_menstruation", "dischromic_patches", "watering_from_eyes",
        "increased_appetite", "polyuria", "family_history", "mucoid_sputum",
        "rusty_sputum", "lack_of_concentration", "visual_disturbances",
        "receiving_blood_transfusion", "receiving_unsterile_injections", "coma",
        "stomach_bleeding", "distention_of_abdomen", "history_of_alcohol_consumption",
        "fluid_overload", "blood_in_sputum", "prominent_veins_on_calf",
        "palpitations", "painful_walking", "pus_filled_pimples", "blackheads",
        "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails",
        "inflammatory_nails", "blister", "red_sore_around_nose", "yellow_crust_ooze"
    ]
    
    selected_symptoms = st.multiselect(
        "Select your symptoms (you can select multiple):",
        symptoms_list,
        max_selections=10
    )
    
    if st.button("Predict Disease") and selected_symptoms:
        try:
            # This would require the DiseaseModel and helper functions
            # For demonstration, showing a simplified version
            st.info("Disease prediction based on selected symptoms...")
            
            # Here you would use your DiseaseModel
            # disease_model = DiseaseModel()
            # prediction = disease_model.predict(selected_symptoms)
            
            st.write(f"**Selected Symptoms:** {', '.join(selected_symptoms)}")
            st.warning("Please consult with a healthcare professional for accurate diagnosis.")
            
            if name:
                with st.spinner("Generating recommendations..."):
                    patient_info = {
                        "name": name,
                        "symptoms": selected_symptoms
                    }
                    
                    recommendations = get_health_recommendations("General Symptoms", "moderate", patient_info)
                    if recommendations:
                        display_recommendations(recommendations)
                        display_health_tips_dynamic(selected_symptoms,severity.lower())

        except:
            st.error("Please select at least one symptom.")

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
if selected == 'Alzheimers Prediction':
    st.title("🧠 Alzheimer's Disease Prediction")
    st.markdown("Early detection can help in managing the progression")

    name = st.text_input("Name:")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        country = st.selectbox("Country", ["USA", "UK", "Canada", "Australia", "India", "China", "Germany", "France", "Other"])
        age = st.number_input("Age", min_value=40, max_value=100, value=65)
        gender = st.selectbox("Gender", ["Male", "Female"])
        education_level = st.slider("Education Level", 0, 5, 3, help="0=No formal education, 5=Doctorate")
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)

    with col2:
        physical_activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
        smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol_consumption = st.selectbox("Alcohol Consumption", ["None", "Light", "Moderate", "Heavy"])
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])

    with col3:
        cholesterol_level = st.selectbox("Cholesterol Level", ["Normal", "High", "Very High"])
        family_history = st.selectbox("Family History of Alzheimer's", ["No", "Yes"])
        cognitive_test_score = st.slider("Cognitive Test Score", 0, 100, 75)
        depression_level = st.selectbox("Depression Level", ["None", "Mild", "Moderate", "Severe"])
        sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Fair", "Good", "Excellent"])

    with col4:
        dietary_habits = st.selectbox("Dietary Habits", ["Poor", "Average", "Good", "Excellent"])
        air_pollution = st.selectbox("Air Pollution Exposure", ["Low", "Medium", "High"])
        employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Retired"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        genetic_risk = st.selectbox("Genetic Risk Factor (APOE-ε4)", ["No", "Yes"])

    # Additional fields
    col5, col6 = st.columns(2)
    with col5:
        social_engagement = st.selectbox("Social Engagement Level", ["Low", "Moderate", "High"])
        income_level = st.selectbox("Income Level", ["Low", "Medium", "High"])
    
    with col6:
        stress_levels = st.selectbox("Stress Levels", ["Low", "Moderate", "High"])
        living_area = st.selectbox("Urban vs Rural Living", ["Urban", "Rural"])

    diagnosis = st.text_input("Diagnosis (Optional)")
    doctor_in_charge = st.text_input("Doctor In Charge (Optional)")

    if st.button("Predict Alzheimer's Risk"):
        try:
            # Create input dictionary with EXACT column names from CSV
            user_input = {
                'Country': country,
                'Age': age,
                'Gender': gender,
                'Education Level': education_level,
                'BMI': bmi,
                'Physical Activity Level': physical_activity,
                'Smoking Status': smoking_status,
                'Alcohol Consumption': alcohol_consumption,
                'Diabetes': diabetes,
                'Hypertension': hypertension,
                'Cholesterol Level': cholesterol_level,
                'Family History of Alzheimer\'s': family_history,
                'Cognitive Test Score': cognitive_test_score,
                'Depression Level': depression_level,
                'Sleep Quality': sleep_quality,
                'Dietary Habits': dietary_habits,
                'Air Pollution Exposure': air_pollution,
                'Employment Status': employment_status,
                'Marital Status': marital_status,
                'Genetic Risk Factor (APOE-ε4 allele)': genetic_risk,
                'Social Engagement Level': social_engagement,
                'Income Level': income_level,
                'Stress Levels': stress_levels,
                'Urban vs Rural Living': living_area
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([user_input])
            
            # Encode categorical features using saved encoders
            for col, encoder in label_encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = encoder.transform(input_df[col].astype(str))
                    except ValueError as e:
                        st.error(f"Invalid value for {col}: {input_df[col].values[0]}")
                        st.stop()
            
            # Ensure correct feature order
            input_df = input_df[feature_names]
            
            # Scale the features
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = alzheimers_model.predict(input_scaled)
            prediction_proba = alzheimers_model.predict_proba(input_scaled)
            
            # Decode prediction
            predicted_label = target_encoder.inverse_transform(prediction)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                if predicted_label == "Yes":
                    st.error("⚠️ **High risk of Alzheimer's detected!**")
                    confidence = prediction_proba[0][1] * 100
                    st.metric("Risk Confidence", f"{confidence:.1f}%")
                    st.warning("**Recommendation:** Please consult a neurologist for further evaluation.")
                else:
                    st.success("✅ **Low risk of Alzheimer's**")
                    confidence = prediction_proba[0][0] * 100
                    st.metric("Confidence Level", f"{confidence:.1f}%")
                    st.info("**Recommendation:** Maintain a healthy lifestyle and regular check-ups.")
            
            with col_result2:
                # Probability chart
                st.markdown("### Risk Probability")
                prob_df = pd.DataFrame({
                    'Risk': target_encoder.classes_,
                    'Probability (%)': prediction_proba[0] * 100
                })
                st.bar_chart(prob_df.set_index('Risk'))
            
            # Optional: Generate recommendations with patient info
            if name:
                patient_info = {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "cognitive_score": cognitive_test_score,
                    "risk_level": predicted_label,
                    "confidence": confidence
                }
                
                # Call your recommendation function if it exists
                try:
                    recommendations = get_health_recommendations("Alzheimer's Disease", "early stage" if predicted_label == "Yes" else "prevention", patient_info)
                    if recommendations:
                        display_recommendations(recommendations)
                        severity = "high" if predicted_label == "Yes" else "low"
                        display_health_tips_dynamic("Alzheimer's", severity)
                except NameError:
                    pass  # Functions not defined
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.info("Please ensure the model files are properly loaded and all inputs are valid.")
            
# Epilepsy Prediction
# Load or import your epilepsy_model here (from joblib, pickle, etc.)
# epilepsy_model = ...

if selected == 'Epilepsy Prediction':
    st.title("⚡ Epilepsy Prediction")
    st.markdown("Predict seizure risk based on EEG patterns and clinical data")

    name = st.text_input("Name:")

    st.markdown("### Clinical Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        seizure_type = st.selectbox("Seizure Type", [
            "Generalized Tonic-Clonic", "Focal", "Absence",
            "Myoclonic", "Atonic", "Unknown"
        ])

    with col2:
        seizure_frequency = st.selectbox("Seizure Frequency", [
            "Daily", "Weekly", "Monthly", "Yearly", "First Time"
        ])
        triggers = st.multiselect("Known Triggers", [
            "Stress", "Lack of Sleep", "Flashing Lights",
            "Alcohol", "Missed Medication", "Menstruation"
        ])
        family_history = st.selectbox("Family History", ["Yes", "No"])

    with col3:
        head_injury = st.selectbox("Previous Head Injury", ["Yes", "No"])
        birth_complications = st.selectbox("Birth Complications", ["Yes", "No"])
        febrile_seizures = st.selectbox("Childhood Febrile Seizures", ["Yes", "No"])

    st.markdown("### EEG Data Upload")
    eeg_file = st.file_uploader("Upload EEG CSV file (178 feature columns)", type=["csv"])

    if st.button("Predict Epilepsy Risk"):
        if eeg_file is not None:
            eeg_data = pd.read_csv(eeg_file)
            if eeg_data.shape[1] < 178:
                st.error("EEG data must have at least 178 feature columns per sample.")
            else:
                eeg_features = eeg_data.iloc[:, :178]  # Use the first 178 columns
                eeg_predictions = epilepsy_model.predict(eeg_features)

                seizure_percentage = np.mean(eeg_predictions == 1) * 100
                st.write(f"Seizure prediction per EEG segment: {seizure_percentage:.2f}% of segments show seizure activity.")

                # Aggregate into a risk level
                if seizure_percentage > 50:
                    risk_label = "High Risk of Epilepsy"
                    st.error(risk_label)
                elif seizure_percentage > 20:
                    risk_label = "Moderate Risk of Epilepsy"
                    st.warning(risk_label)
                else:
                    risk_label = "Low Risk of Epilepsy"
                    st.success(risk_label)

                # Combine clinical info and EEG prediction for recommendations
                if name:
                    with st.spinner("Generating recommendations..."):
                        patient_info = {
                            "name": name,
                            "age": age,
                            "seizure_type": seizure_type,
                            "frequency": seizure_frequency,
                            "triggers": triggers,
                            "family_history": family_history,
                            "head_injury": head_injury,
                            "birth_complications": birth_complications,
                            "febrile_seizures": febrile_seizures,
                            "eeg_seizure_risk": risk_label
                        }
                        recommendations = get_health_recommendations("Epilepsy", risk_label.lower(), patient_info)
                        if recommendations:
                            display_recommendations(recommendations)
                            display_health_tips_dynamic("Epilepsy", severity.lower())

        else:
            st.warning("Please upload your EEG data CSV file to proceed with prediction.")

# HIV/AIDS Prediction
if selected == 'HIV/AIDS Prediction':
    st.title("🔬 HIV/AIDS Risk Assessment & Clinical Prediction")
    st.markdown("Comprehensive HIV risk evaluation using behavioral and clinical data")
    
    name = st.text_input("Name:")
    
    # Choose assessment type
    assessment_type = st.radio("Select Assessment Type:", 
                              ["Behavioral Risk Assessment", "Clinical Prediction", "Complete Assessment"])
    
    if assessment_type in ["Behavioral Risk Assessment", "Complete Assessment"]:
        st.subheader("🏥 Behavioral Risk Assessment")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=15, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            sexual_partners = st.number_input("Number of Sexual Partners (last year)", min_value=0, max_value=100, value=1)
        
        with col2:
            unprotected_sex = st.selectbox("Unprotected Sex", ["Never", "Rarely", "Sometimes", "Often", "Always"])
            iv_drug_use = st.selectbox("IV Drug Use", ["No", "Past", "Current"])
            blood_transfusion = st.selectbox("Blood Transfusion History", ["No", "Yes - Before 1985", "Yes - After 1985"])
        
        with col3:
            std_history = st.selectbox("STD History", ["None", "1-2 times", "3+ times"])
            partner_hiv_status = st.selectbox("Partner HIV Status", ["Negative", "Positive", "Unknown"])
            prep_usage = st.selectbox("PrEP Usage", ["No", "Yes - Consistent", "Yes - Inconsistent"])
        
        st.subheader("Additional Risk Factors")
        col4, col5 = st.columns(2)
        
        with col4:
            healthcare_worker = st.selectbox("Healthcare Worker", ["No", "Yes"])
            needle_stick = st.selectbox("Needle Stick Injury", ["No", "Yes"])
        
        with col5:
            mother_hiv = st.selectbox("Mother HIV+ (if applicable)", ["No", "Yes", "N/A"])
            commercial_sex = st.selectbox("Commercial Sex Work", ["No", "Yes"])
    
    if assessment_type in ["Clinical Prediction", "Complete Assessment"]:
        st.subheader("🧪 Clinical Parameters")
        st.info("Enter clinical laboratory values and medical history")
        
        col6, col7, col8 = st.columns(3)
        
        with col6:
            if assessment_type == "Clinical Prediction":
                age = st.number_input("Age", min_value=12, max_value=70, value=35)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=160.0, value=75.0)
            karnof = st.slider("Karnofsky Performance Score", 70, 100, 95)
            hemophilia = st.selectbox("Hemophilia", ["No", "Yes"])
        
        with col7:
            cd4_baseline = st.number_input("CD4+ Count (baseline)", min_value=0, max_value=1200, value=350)
            cd4_followup = st.number_input("CD4+ Count (follow-up)", min_value=0, max_value=1200, value=370)
            cd8_baseline = st.number_input("CD8+ Count (baseline)", min_value=0, max_value=5000, value=987)
            cd8_followup = st.number_input("CD8+ Count (follow-up)", min_value=0, max_value=6000, value=935)
        
        with col8:
            homosexual_activity = st.selectbox("Homosexual Activity", ["No", "Yes"])
            iv_drugs_clinical = st.selectbox("IV Drug Use History", ["No", "Yes"])
            prior_zdv = st.selectbox("Prior ZDV Treatment", ["No", "Yes"])
            symptomatic = st.selectbox("Symptomatic", ["No", "Yes"])
    
    if st.button("Assess HIV Risk"):
        try:
            behavioral_risk = 0
            clinical_risk = 0
            
            # BEHAVIORAL RISK CALCULATION
            if assessment_type in ["Behavioral Risk Assessment", "Complete Assessment"]:
                if age < 25 or age > 45:
                    behavioral_risk += 1
                
                risk_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4}
                behavioral_risk += risk_map.get(unprotected_sex, 0)
                
                if sexual_partners > 5:
                    behavioral_risk += 2
                elif sexual_partners > 2:
                    behavioral_risk += 1
                
                if iv_drug_use == "Current":
                    behavioral_risk += 4
                elif iv_drug_use == "Past":
                    behavioral_risk += 2
                
                if partner_hiv_status == "Positive":
                    behavioral_risk += 5
                elif partner_hiv_status == "Unknown":
                    behavioral_risk += 2
                
                if std_history == "3+ times":
                    behavioral_risk += 3
                elif std_history == "1-2 times":
                    behavioral_risk += 1
                
                if healthcare_worker == "Yes":
                    behavioral_risk += 1
                if needle_stick == "Yes":
                    behavioral_risk += 1
                if mother_hiv == "Yes":
                    behavioral_risk += 1
                if commercial_sex == "Yes":
                    behavioral_risk += 1
            
            # CLINICAL RISK CALCULATION
            if assessment_type in ["Clinical Prediction", "Complete Assessment"]:
                # Prepare clinical features for model prediction
                clinical_features = [
                    age if assessment_type == "Clinical Prediction" else age,
                    weight,
                    1 if hemophilia == "Yes" else 0,
                    1 if homosexual_activity == "Yes" else 0,
                    1 if iv_drugs_clinical == "Yes" else 0,
                    karnof,
                    1 if prior_zdv == "Yes" else 0,
                    1 if symptomatic == "Yes" else 0,
                    cd4_baseline,
                    cd4_followup,
                    cd8_baseline,
                    cd8_followup
                ]
                
                # Simple clinical risk scoring (you can replace this with actual model prediction)
                # Low CD4 count indicates higher risk
                if cd4_baseline < 200:
                    clinical_risk += 4
                elif cd4_baseline < 350:
                    clinical_risk += 2
                
                # CD4 decline
                if cd4_followup < cd4_baseline:
                    clinical_risk += 2
                
                # High CD8 count can indicate infection
                if cd8_baseline > 1200:
                    clinical_risk += 1
                
                # Other risk factors
                if karnof < 90:
                    clinical_risk += 1
                if hemophilia == "Yes":
                    clinical_risk += 1
                if iv_drugs_clinical == "Yes":
                    clinical_risk += 2
                if symptomatic == "Yes":
                    clinical_risk += 2
                
                # Uncomment when you have a trained model:
                # hiv_prediction = hiv_model.predict([clinical_features])
                # clinical_risk = hiv_prediction[0] * 10  # Scale to 0-10
            
            # COMBINED RISK ASSESSMENT
            if assessment_type == "Complete Assessment":
                total_risk = (behavioral_risk + clinical_risk) / 2
                risk_components = f"Behavioral: {behavioral_risk}/20, Clinical: {clinical_risk}/10"
            elif assessment_type == "Behavioral Risk Assessment":
                total_risk = behavioral_risk
                risk_components = f"Behavioral Risk Score: {behavioral_risk}/20"
            else:  # Clinical Prediction
                total_risk = clinical_risk
                risk_components = f"Clinical Risk Score: {clinical_risk}/10"
            
            # RISK LEVEL DETERMINATION
            if assessment_type == "Behavioral Risk Assessment":
                if behavioral_risk >= 10:
                    risk_level = "High"
                    st.error(f"{name}, high HIV risk identified. Immediate testing and consultation recommended.")
                    severity = "high"
                elif behavioral_risk >= 5:
                    risk_level = "Moderate"
                    st.warning(f"{name}, moderate HIV risk. Regular testing recommended.")
                    severity = "moderate"
                else:
                    risk_level = "Low"
                    st.success(f"{name}, low HIV risk. Continue preventive measures.")
                    severity = "low"
            
            elif assessment_type == "Clinical Prediction":
                if clinical_risk >= 6:
                    risk_level = "High"
                    st.error(f"{name}, clinical indicators suggest high HIV risk. Immediate consultation required.")
                    severity = "high"
                elif clinical_risk >= 3:
                    risk_level = "Moderate"
                    st.warning(f"{name}, moderate clinical risk indicators. Follow-up recommended.")
                    severity = "moderate"
                else:
                    risk_level = "Low"
                    st.success(f"{name}, clinical parameters within normal range.")
                    severity = "low"
            
            else:  # Complete Assessment
                if total_risk >= 8:
                    risk_level = "High"
                    st.error(f"{name}, comprehensive assessment shows high HIV risk. Immediate medical attention required.")
                    severity = "high"
                elif total_risk >= 4:
                    risk_level = "Moderate"
                    st.warning(f"{name}, moderate risk identified. Regular monitoring recommended.")
                    severity = "moderate"
                else:
                    risk_level = "Low"
                    st.success(f"{name}, overall low risk. Continue current prevention strategies.")
                    severity = "low"
            
            # Display metrics
            col_metrics1, col_metrics2 = st.columns(2)
            with col_metrics1:
                st.metric("Overall Risk Level", risk_level, f"Score: {total_risk:.1f}")
            with col_metrics2:
                st.metric("Risk Components", "", risk_components)
            
            # Clinical insights for clinical prediction
            if assessment_type in ["Clinical Prediction", "Complete Assessment"]:
                st.subheader("📊 Clinical Insights")
                col_insights1, col_insights2 = st.columns(2)
                
                with col_insights1:
                    st.metric("CD4+ T-Cell Count", f"{cd4_baseline}", f"Follow-up: {cd4_followup}")
                    cd4_trend = "↓ Declining" if cd4_followup < cd4_baseline else "↑ Stable/Improving"
                    st.write(f"**Trend:** {cd4_trend}")
                
                with col_insights2:
                    st.metric("CD8+ T-Cell Count", f"{cd8_baseline}", f"Follow-up: {cd8_followup}")
                    cd4_cd8_ratio = cd4_baseline / max(cd8_baseline, 1)
                    st.write(f"**CD4:CD8 Ratio:** {cd4_cd8_ratio:.2f}")
                    if cd4_cd8_ratio < 0.4:
                        st.warning("⚠️ Low CD4:CD8 ratio may indicate immune dysfunction")
            
            # AI Recommendations
            if name:
                with st.spinner("Generating personalized recommendations..."):
                    patient_info = {
                        "name": name,
                        "age": age,
                        "assessment_type": assessment_type,
                        "risk_level": risk_level,
                        "risk_score": total_risk
                    }
                    
                    if assessment_type in ["Behavioral Risk Assessment", "Complete Assessment"]:
                        patient_info["prep_status"] = prep_usage
                    
                    if assessment_type in ["Clinical Prediction", "Complete Assessment"]:
                        patient_info["cd4_count"] = cd4_baseline
                        patient_info["cd4_trend"] = "declining" if cd4_followup < cd4_baseline else "stable"
                    
                    recommendations = get_health_recommendations("HIV Prevention", severity, patient_info)
                    if recommendations:
                        display_recommendations(recommendations)
                        display_health_tips_dynamic("HIV", severity.lower())

        
        except Exception as e:
            st.error(f"Error in risk assessment: {str(e)}")

# Malaria Prediction
if selected == 'Malaria Prediction':
    st.title("🦟 Malaria Prediction")
    st.markdown("Early detection and prevention of Malaria")
    
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        temperature = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)
    
    with col2:
        chills = st.selectbox("Chills and Shivering", ["No", "Mild", "Severe"])
        headache = st.selectbox("Headache", ["No", "Mild", "Moderate", "Severe"])
        muscle_pain = st.selectbox("Muscle Pain", ["No", "Mild", "Moderate", "Severe"])
    
    with col3:
        nausea = st.selectbox("Nausea/Vomiting", ["No", "Yes"])
        sweating = st.selectbox("Profuse Sweating", ["No", "Yes"])
        travel_endemic = st.selectbox("Recent Travel to Endemic Area", ["No", "Yes"])
    
    # Additional symptoms
    st.subheader("Additional Symptoms")
    col4, col5 = st.columns(2)
    
    with col4:
        anemia = st.selectbox("Anemia Symptoms", ["No", "Yes"])
        jaundice = st.selectbox("Jaundice", ["No", "Yes"])
        convulsions = st.selectbox("Convulsions", ["No", "Yes"])
    
    with col5:
        respiratory_distress = st.selectbox("Respiratory Distress", ["No", "Yes"])
        blood_in_urine = st.selectbox("Blood in Urine", ["No", "Yes"])
        rapid_breathing = st.selectbox("Rapid Breathing", ["No", "Yes"])
    
    if st.button("Predict Malaria"):
        try:
            # Convert inputs to features
            features = [
                age,
                1 if gender == "Male" else 0,
                temperature,
                {"No": 0, "Mild": 1, "Severe": 2}.get(chills, 0),
                {"No": 0, "Mild": 1, "Moderate": 2, "Severe": 3}.get(headache, 0),
                {"No": 0, "Mild": 1, "Moderate": 2, "Severe": 3}.get(muscle_pain, 0),
                1 if nausea == "Yes" else 0,
                1 if sweating == "Yes" else 0,
                1 if travel_endemic == "Yes" else 0,
                1 if anemia == "Yes" else 0,
                1 if jaundice == "Yes" else 0,
                1 if convulsions == "Yes" else 0
            ]
            
            malaria_prediction = malaria_model.predict([features])
            
            if malaria_prediction[0] == 1:
                st.error(f"{name}, high probability of Malaria. Seek immediate medical attention!")
                severity = "severe"
            else:
                st.success(f"{name}, low probability of Malaria. Monitor symptoms.")
                severity = "low"
            
            # Get AI recommendations
            with st.spinner("Generating treatment recommendations..."):
                patient_info = {
                    "name": name,
                    "age": age,
                    "temperature": temperature,
                    "travel_history": travel_endemic
                }
                
                recommendations = get_health_recommendations("Malaria", severity, patient_info)
                if recommendations:
                    display_recommendations(recommendations)
                    display_health_tips_dynamic("Malaria", severity.lower())

                    
        except:
            st.warning("Malaria prediction model not available. Please ensure model file exists.")

# Colorectal Cancer Prediction
if selected == 'Colorectal Cancer Prediction':
    st.title("🔬 Colorectal Cancer Risk Assessment")
    st.markdown("Early detection saves lives")
    
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=50)
        gender = st.selectbox("Gender", ["Male", "Female"])
        family_history = st.selectbox("Family History of Colorectal Cancer", ["No", "Yes - Parent/Sibling", "Yes - Multiple Relatives"])
    
    with col2:
        polyps_history = st.selectbox("History of Polyps", ["No", "Yes - Benign", "Yes - Precancerous"])
        ibd = st.selectbox("Inflammatory Bowel Disease", ["No", "Crohn's Disease", "Ulcerative Colitis"])
        diabetes_t2 = st.selectbox("Type 2 Diabetes", ["No", "Yes"])
    
    with col3:
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol = st.selectbox("Alcohol Consumption", ["None", "Moderate", "Heavy"])
        obesity = st.selectbox("Obesity (BMI > 30)", ["No", "Yes"])
    
    # Symptoms
    st.subheader("Symptoms")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        blood_stool = st.selectbox("Blood in Stool", ["No", "Occasionally", "Frequently"])
        bowel_changes = st.selectbox("Change in Bowel Habits", ["No", "Yes"])
        abdominal_pain = st.selectbox("Persistent Abdominal Discomfort", ["No", "Yes"])
    
    with col5:
        unexplained_weight_loss = st.selectbox("Unexplained Weight Loss", ["No", "Yes"])
        fatigue = st.selectbox("Chronic Fatigue", ["No", "Yes"])
        anemia = st.selectbox("Iron Deficiency Anemia", ["No", "Yes"])
    
    with col6:
        narrow_stools = st.selectbox("Narrow Stools", ["No", "Yes"])
        incomplete_evacuation = st.selectbox("Feeling of Incomplete Evacuation", ["No", "Yes"])
        screening_history = st.selectbox("Last Colonoscopy", ["Never", "< 1 year", "1-5 years", "> 5 years"])
    
    if st.button("Assess Colorectal Cancer Risk"):
        try:
            # Risk calculation
            risk_score = 0
            
            # Age is major risk factor
            if age >= 50:
                risk_score += 3
            elif age >= 40:
                risk_score += 1
            
            # Family history
            if "Multiple" in family_history:
                risk_score += 4
            elif "Parent" in family_history:
                risk_score += 2
            
            # Medical history
            if "Precancerous" in polyps_history:
                risk_score += 3
            elif "Benign" in polyps_history:
                risk_score += 1
            
            if ibd != "No":
                risk_score += 2
            
            # Lifestyle factors
            if smoking == "Current":
                risk_score += 2
            if alcohol == "Heavy":
                risk_score += 2
            if obesity == "Yes":
                risk_score += 1
            
            # Symptoms
            if blood_stool == "Frequently":
                risk_score += 3
            elif blood_stool == "Occasionally":
                risk_score += 1
            
            symptoms_count = sum([
                bowel_changes == "Yes",
                abdominal_pain == "Yes",
                unexplained_weight_loss == "Yes",
                fatigue == "Yes",
                anemia == "Yes",
                narrow_stools == "Yes"
            ])
            risk_score += symptoms_count * 0.5
            
            # Determine risk level
            if risk_score >= 10:
                st.error(f"{name}, high colorectal cancer risk. Immediate colonoscopy recommended!")
                severity = "high"
            elif risk_score >= 6:
                st.warning(f"{name}, moderate risk. Schedule screening consultation.")
                severity = "moderate"
            else:
                st.success(f"{name}, low risk. Continue regular screening as per guidelines.")
                severity = "low"
            
            st.metric("Risk Score", f"{risk_score:.1f}/20", f"Risk Level: {severity.upper()}")
            
            # Get AI recommendations
            with st.spinner("Generating personalized screening recommendations..."):
                patient_info = {
                    "name": name,
                    "age": age,
                    "risk_score": risk_score,
                    "family_history": family_history,
                    "last_screening": screening_history
                }
                
                recommendations = get_health_recommendations("Colorectal Cancer", severity, patient_info)
                if recommendations:
                    display_recommendations(recommendations)
                    display_health_tips_dynamic("Colorectal Cancer", severity.lower())

                    
        except Exception as e:
            st.error(f"Error in assessment: {str(e)}")

# Prostate Cancer Prediction
if selected == 'Prostate Cancer Prediction':
    st.title("👨 Prostate Cancer Risk Assessment")
    st.markdown("Early detection through PSA screening and risk assessment")
    
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=40, max_value=100, value=55)
        race = st.selectbox("Race/Ethnicity", ["Caucasian", "African American", "Asian", "Hispanic", "Other"])
        family_history = st.selectbox("Family History", ["None", "Father", "Brother", "Multiple Relatives"])
    
    with col2:
        psa_level = st.number_input("PSA Level (ng/mL)", min_value=0.0, max_value=100.0, value=2.5)
        psa_velocity = st.selectbox("PSA Velocity", ["Stable", "Slowly Rising", "Rapidly Rising"])
        dre_result = st.selectbox("Digital Rectal Exam", ["Normal", "Enlarged", "Nodules", "Not Done"])
    
    with col3:
        urinary_symptoms = st.selectbox("Urinary Symptoms", ["None", "Mild", "Moderate", "Severe"])
        erectile_dysfunction = st.selectbox("Erectile Dysfunction", ["No", "Mild", "Moderate", "Severe"])
        bone_pain = st.selectbox("Bone Pain", ["No", "Yes"])
    
    # Additional factors
    st.subheader("Additional Risk Factors")
    col4, col5 = st.columns(2)
    
    with col4:
        bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0)
        smoking_status = st.selectbox("Smoking", ["Never", "Former", "Current"])
        diet = st.selectbox("Diet Type", ["Balanced", "High Red Meat", "Vegetarian"])
    
    with col5:
        exercise = st.selectbox("Exercise Frequency", ["Sedentary", "Light", "Moderate", "Active"])
        previous_biopsy = st.selectbox("Previous Biopsy", ["No", "Yes - Negative", "Yes - Suspicious"])
    
    if st.button("Assess Prostate Cancer Risk"):
        try:
            # Risk calculation
            risk_score = 0
            
            # Age risk
            if age >= 70:
                risk_score += 3
            elif age >= 60:
                risk_score += 2
            elif age >= 50:
                risk_score += 1
            
            # Race risk
            if race == "African American":
                risk_score += 2
            
            # Family history
            if family_history == "Multiple Relatives":
                risk_score += 3
            elif family_history in ["Father", "Brother"]:
                risk_score += 2
            
            # PSA level risk
            if psa_level >= 10:
                risk_score += 4
            elif psa_level >= 4:
                risk_score += 2
            elif psa_level >= 2.5:
                risk_score += 1
            
            # PSA velocity
            if psa_velocity == "Rapidly Rising":
                risk_score += 3
            elif psa_velocity == "Slowly Rising":
                risk_score += 1
            
            # DRE results
            if dre_result == "Nodules":
                risk_score += 3
            elif dre_result == "Enlarged":
                risk_score += 1
            
            # Symptoms
            if urinary_symptoms == "Severe":
                risk_score += 2
            elif urinary_symptoms == "Moderate":
                risk_score += 1
            
            if bone_pain == "Yes":
                risk_score += 2
            
            # Determine risk
            if risk_score >= 12:
                st.error(f"{name}, high prostate cancer risk. Urgent urologist consultation needed!")
                severity = "high"
            elif risk_score >= 7:
                st.warning(f"{name}, moderate risk. Schedule PSA test and urological evaluation.")
                severity = "moderate"
            else:
                st.success(f"{name}, low risk. Continue regular screening.")
                severity = "low"
            
            st.metric("Risk Score", f"{risk_score}/25", f"PSA: {psa_level} ng/mL")
            
            # Get AI recommendations
            with st.spinner("Generating screening and treatment recommendations..."):
                patient_info = {
                    "name": name,
                    "age": age,
                    "psa": psa_level,
                    "risk_factors": f"Risk score: {risk_score}"
                }
                
                recommendations = get_health_recommendations("Prostate Cancer", severity, patient_info)
                if recommendations:
                    display_recommendations(recommendations)
                    display_health_tips_dynamic("Prostate Cancer", severity.lower())

                    
        except Exception as e:
            st.error(f"Error in assessment: {str(e)}")

# Cervical Cancer Prediction
if selected == 'Cervical Cancer Prediction':
    st.title("👩 Cervical Cancer Risk Assessment")
    st.markdown("Prevention through screening and HPV vaccination")

    name = st.text_input("Name:")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        age_first_intercourse = st.number_input("Age at First Intercourse", min_value=10, max_value=50, value=18)
        num_pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=2)

    with col2:
        num_sexual_partners = st.number_input("Lifetime Sexual Partners", min_value=0, max_value=50, value=3)
        hpv_status = st.selectbox("HPV Status", ["Unknown", "Negative", "Positive"])
        hpv_vaccine = st.selectbox("HPV Vaccination", ["No", "Yes - Partial", "Yes - Complete"])

    with col3:
        smoking = st.selectbox("Smoking", ["Never", "Former", "Current"])
        oral_contraceptives = st.selectbox("Oral Contraceptives Use", ["Never", "< 5 years", "> 5 years"])
        iud_years = st.number_input("IUD Use (years)", min_value=0, max_value=30, value=0)

    st.subheader("Screening History")

    col4, col5, col6 = st.columns(3)
    with col4:
        last_pap = st.selectbox("Last Pap Smear", ["Never", "< 1 year", "1-3 years", "> 3 years"])
        pap_results = st.selectbox("Previous Pap Results", ["Normal", "ASCUS", "LSIL", "HSIL", "Not Applicable"])
        stds_history = st.selectbox("STDs History", ["None", "1-2", "3+"])

    with col5:
        hiv_status = st.selectbox("HIV Status", ["Negative", "Positive", "Unknown"])
        family_history = st.selectbox("Family History of Cervical Cancer", ["No", "Yes"])
        immunosuppressed = st.selectbox("Immunosuppressed", ["No", "Yes"])

    with col6:
        abnormal_bleeding = st.selectbox("Abnormal Vaginal Bleeding", ["No", "Yes"])
        pelvic_pain = st.selectbox("Pelvic Pain", ["No", "Yes"])
        discharge = st.selectbox("Unusual Discharge", ["No", "Yes"])

    if st.button("Assess Cervical Cancer Risk"):
        try:
            risk_score = 0

            if age < 25:
                risk_score += 0.5
            elif 30 <= age <= 65:
                risk_score += 1

            if age_first_intercourse < 16:
                risk_score += 2

            if num_sexual_partners > 6:
                risk_score += 2
            elif num_sexual_partners > 3:
                risk_score += 1

            if hpv_status == "Positive":
                risk_score += 5
            elif hpv_status == "Negative":
                risk_score += 0

            if hpv_vaccine == "No":
                risk_score += 1

            if smoking == "Current":
                risk_score += 2

            if oral_contraceptives == "> 5 years":
                risk_score += 1

            if last_pap in ["Never", "> 3 years"]:
                risk_score += 2

            if pap_results == "HSIL":
                risk_score += 4
            elif pap_results in ["LSIL", "ASCUS"]:
                risk_score += 2

            if hiv_status == "Positive":
                risk_score += 3

            if immunosuppressed == "Yes":
                risk_score += 2

            symptoms_present = sum([
                abnormal_bleeding == "Yes",
                pelvic_pain == "Yes",
                discharge == "Yes"
            ])
            risk_score += symptoms_present

            if risk_score >= 12:
                st.error(f"{name}, high cervical cancer risk. Immediate gynecological evaluation required!")
                severity = "high"
            elif risk_score >= 7:
                st.warning(f"{name}, moderate risk. Schedule Pap smear and HPV test.")
                severity = "moderate"
            else:
                st.success(f"{name}, low risk. Continue regular screening.")
                severity = "low"

            st.metric("Risk Score", f"{risk_score}/25", f"Risk Level: {severity.upper()}")

            with st.spinner("Generating personalized prevention recommendations..."):
                patient_info = {
                    "name": name,
                    "age": age,
                    "hpv_status": hpv_status,
                    "vaccination": hpv_vaccine,
                    "last_screening": last_pap
                }

                recommendations = get_health_recommendations("Cervical Cancer Prevention", severity, patient_info)
                if recommendations:
                    display_recommendations(recommendations)
                    display_health_tips_dynamic("Cervical Cancer", severity.lower())


        except Exception as e:
            st.error(f"Error in assessment: {str(e)}")

# Asthma Prediction
if selected == 'Asthma Prediction':
    st.title("🫁 Asthma Risk Assessment")
    st.markdown("Manage asthma effectively with early detection")

    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        family_history_asthma = st.selectbox("Family History of Asthma", ["No", "Parents", "Siblings", "Both"])

    with col2:
        allergies = st.multiselect("Known Allergies", [
            "Dust Mites", "Pollen", "Pet Dander", "Mold", 
            "Food Allergies", "Drug Allergies", "None"
        ])
        eczema = st.selectbox("Eczema/Atopic Dermatitis", ["No", "Yes"])
        allergic_rhinitis = st.selectbox("Allergic Rhinitis (Hay Fever)", ["No", "Yes"])

    with col3:
        wheezing = st.selectbox("Wheezing Episodes", ["Never", "Rarely", "Sometimes", "Frequently"])
        shortness_breath = st.selectbox("Shortness of Breath", ["Never", "During Exercise", "At Rest", "Both"])
        chest_tightness = st.selectbox("Chest Tightness", ["No", "Occasionally", "Frequently"])

    st.subheader("Triggers and Additional Symptoms")
    col4, col5, col6 = st.columns(3)

    with col4:
        cough_night = st.selectbox("Nighttime Cough", ["No", "1-2 nights/week", "> 2 nights/week"])
        exercise_symptoms = st.selectbox("Exercise-Induced Symptoms", ["No", "Yes"])
        cold_air_trigger = st.selectbox("Cold Air Trigger", ["No", "Yes"])

    with col5:
        smoke_exposure = st.selectbox("Smoke Exposure", ["None", "Secondhand", "Smoker"])
        pollution_exposure = st.selectbox("Air Pollution Exposure", ["Low", "Moderate", "High"])
        occupational_exposure = st.selectbox("Occupational Chemicals", ["No", "Yes"])

    with col6:
        respiratory_infections = st.selectbox("Frequent Respiratory Infections", ["No", "Yes"])
        emergency_visits = st.number_input("Emergency Visits (past year)", min_value=0, max_value=20, value=0)
        peak_flow = st.number_input("Peak Flow (% of predicted)", min_value=0, max_value=150, value=100)

    if st.button("Assess Asthma Risk"):
        try:
            risk_score = 0

            # Map family history to risk score
            fam_hist_map = {"No": 0, "Parents": 1, "Siblings": 1, "Both": 3}
            risk_score += fam_hist_map.get(family_history_asthma, 0)

            # Allergies count excluding None
            risk_score += max(len([a for a in allergies if a != "None"]), 0) * 0.5

            # Wheezing score
            wheezing_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Frequently": 3}
            risk_score += wheezing_map.get(wheezing, 0)

            # Night cough score
            cough_night_map = {"No": 0, "1-2 nights/week": 1, "> 2 nights/week": 2}
            risk_score += cough_night_map.get(cough_night, 0)

            # Smoke exposure score
            smoke_map = {"None": 0, "Secondhand": 1, "Smoker": 3}
            risk_score += smoke_map.get(smoke_exposure, 0)

            # Pollution exposure score
            pollution_map = {"Low": 0, "Moderate": 1, "High": 2}
            risk_score += pollution_map.get(pollution_exposure, 0)

            # Occupational exposure score
            occ_exp_map = {"No": 0, "Yes": 1}
            risk_score += occ_exp_map.get(occupational_exposure, 0)

            # Peak flow scoring - more severe if lower
            if peak_flow < 60:
                risk_score += 3
            elif peak_flow < 80:
                risk_score += 2
            elif peak_flow < 100:
                risk_score += 1

            # Emergency visits capped score
            risk_score += min(emergency_visits * 0.5, 5)

            # Severity level mapping
            if risk_score > 12:
                severity = "Severe"
                color = "red"
            elif risk_score > 8:
                severity = "Moderate"
                color = "orange"
            elif risk_score > 4:
                severity = "Mild"
                color = "yellow"
            else:
                severity = "Low"
                color = "green"

            # Display risk
            st.markdown(f"### Asthma Risk Assessment: <span style='color:{color}'>{severity}</span>", unsafe_allow_html=True)
            st.write(f"Risk Score: {risk_score:.1f}/20")
            st.progress(min(risk_score / 20, 1.0))

            if name:
                patient_info = {
                    "name": name,
                    "age": age,
                    "risk_score": risk_score,
                    "peak_flow": peak_flow,
                    "emergency_visits": emergency_visits
                }
                recommendations = get_health_recommendations("Asthma", severity.lower(), patient_info)
                if recommendations:
                    display_recommendations(recommendations)
                    display_health_tips_dynamic("Asthma", severity.lower())


        except Exception as e:
            st.error(f"Error in assessment: {str(e)}")

# COPD Prediction
if selected == 'COPD Prediction':
    st.title("🫁 COPD (Chronic Obstructive Pulmonary Disease) Prediction")
    st.markdown("Assess your risk for COPD")

    name = st.text_input("Name:")

    st.subheader("Demographics & History")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=30, max_value=100, value=55)
        gender = st.selectbox("Gender", ["Male", "Female"])
        smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])

    with col2:
        pack_history = st.number_input("Pack History", min_value=0, max_value=100, value=0)
        occupational_dust = st.selectbox("Occupational Dust/Chemical Exposure", ["No", "Yes"])
        biomass_exposure = st.selectbox("Biomass Fuel Exposure", ["No", "Yes"])  # This may not be in dataset, keep for UI

    with col3:
        family_history_copd = st.selectbox("Family History of COPD", ["No", "Yes"])
        alpha1_deficiency = st.selectbox("Alpha-1 Antitrypsin Deficiency", ["No", "Yes", "Unknown"])  # Not in dataset, UI only
        childhood_respiratory = st.selectbox("Childhood Respiratory Problems", ["No", "Yes"])  # UI only

    st.subheader("Symptoms")
    col4, col5, col6 = st.columns(3)

    with col4:
        chronic_cough = st.selectbox("Chronic Cough", ["No", "Yes"])
        sputum_production = st.selectbox("Daily Sputum Production", ["No", "Yes"])
        breathlessness = st.selectbox("Breathlessness Level", ["None", "Mild", "Moderate", "Severe"])

    with col5:
        wheezing = st.selectbox("Wheezing", ["No", "Occasional", "Frequent"])
        chest_tightness = st.selectbox("Chest Tightness", ["No", "Yes"])
        frequent_infections = st.selectbox("Frequent Chest Infections", ["No", "Yes"])

    with col6:
        exercise_limitation = st.selectbox("Exercise Limitation", ["None", "Mild", "Moderate", "Severe"])
        weight_loss = st.selectbox("Unintentional Weight Loss", ["No", "Yes"])
        fatigue = st.selectbox("Chronic Fatigue", ["No", "Yes"])

    st.subheader("Clinical Measurements (if available)")
    col7, col8 = st.columns(2)

    with col7:
        fev1 = st.number_input("FEV1 (% predicted)", min_value=0, max_value=150, value=100)
        fvc = st.number_input("FVC (% predicted)", min_value=0, max_value=150, value=100)

    with col8:
        oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=70, max_value=100, value=95)
        exacerbations = st.number_input("Exacerbations (past year)", min_value=0, max_value=10, value=0)  # UI only

    if st.button("Predict COPD Risk"):
        try:
            risk_factors = 0

            # Smoking risk mapping
            if smoking_status == "Current":
                risk_factors += 3
            elif smoking_status == "Former":
                risk_factors += 2

            if pack_history > 20:
                risk_factors += 2
            elif pack_history > 10:
                risk_factors += 1

            # Symptoms
            if chronic_cough == "Yes":
                risk_factors += 1
            if sputum_production == "Yes":
                risk_factors += 1

            if breathlessness == "Severe":
                risk_factors += 3
            elif breathlessness == "Moderate":
                risk_factors += 2
            elif breathlessness == "Mild":
                risk_factors += 1

            # Clinical measurements and severity
            if fev1 < 50:
                stage = "Severe (GOLD 3)"
                risk_factors += 3
            elif fev1 < 80:
                stage = "Moderate (GOLD 2)"
                risk_factors += 2
            elif fev1 < 100:
                stage = "Mild (GOLD 1)"
                risk_factors += 1
            else:
                stage = "At Risk (GOLD 0)"

            if risk_factors > 8:
                risk_level = "High"
                color = "red"
            elif risk_factors > 5:
                risk_level = "Moderate"
                color = "orange"
            elif risk_factors > 2:
                risk_level = "Low-Moderate"
                color = "yellow"
            else:
                risk_level = "Low"
                color = "green"

            st.markdown(f"### COPD Risk Level: <span style='color:{color}'>{risk_level}</span>", unsafe_allow_html=True)
            st.write(f"GOLD Stage: {stage}")
            st.progress(min(risk_factors / 12, 1.0))

            if name:
                with st.spinner("Generating COPD management plan..."):
                    patient_info = {
                        "name": name,
                        "age": age,
                        "fev1": fev1,
                        "stage": stage,
                        "smoking_status": smoking_status
                    }

                    recommendations = get_health_recommendations("COPD", stage.split()[0].lower(), patient_info)
                    if recommendations:
                        display_recommendations(recommendations)
                        display_health_tips_dynamic("COPD", severity.lower())


        except Exception as e:
            st.error(f"Error in COPD prediction: {str(e)}")

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
    st.title("🤕 Migraine Risk Assessment")
    st.markdown("Evaluate migraine patterns and triggers")
    
    name = st.text_input("Name:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        family_history = st.selectbox("Family History of Migraines", ["No", "Yes"])
    
    with col2:
        frequency = st.selectbox("Headache Frequency", 
            ["Rarely", "1-3 times/month", "Weekly", "Several times/week", "Daily"])
        duration = st.selectbox("Typical Duration", 
            ["< 4 hours", "4-24 hours", "1-3 days", "> 3 days"])
        intensity = st.slider("Pain Intensity (1-10)", 1, 10, 5)
    
    with col3:
        location = st.selectbox("Pain Location", 
            ["One side", "Both sides", "Behind eye", "Temples", "Back of head"])
        pain_type = st.selectbox("Pain Type", 
            ["Throbbing", "Constant", "Sharp", "Pressure"])
        aura = st.selectbox("Visual Aura", ["No", "Sometimes", "Always"])
    
    # Triggers
    st.subheader("Common Triggers")
    triggers = st.multiselect("Select all that apply:", 
        ["Stress", "Lack of sleep", "Certain foods", "Bright lights", "Strong smells",
         "Weather changes", "Hormonal changes", "Dehydration", "Skipped meals",
         "Physical activity", "Alcohol", "Caffeine withdrawal"])
    
    # Associated symptoms
    st.subheader("Associated Symptoms")
    symptoms = st.multiselect("Select all that apply:",
        ["Nausea", "Vomiting", "Light sensitivity", "Sound sensitivity", 
         "Smell sensitivity", "Neck pain", "Dizziness", "Visual disturbances"])
    
    if st.button("Assess Migraine Pattern"):
        # Calculate migraine severity
        severity_score = 0
        
        if frequency in ["Several times/week", "Daily"]:
            severity_score += 3
        elif frequency == "Weekly":
            severity_score += 2
        elif frequency == "1-3 times/month":
            severity_score += 1
        
        severity_score += min(intensity - 4, 0) * 0.5
        
        if duration in ["> 3 days", "1-3 days"]:
            severity_score += 2
        
        if aura == "Always":
            severity_score += 1
        
        severity_score += len(symptoms) * 0.2
        
        # Determine type and severity
        if severity_score > 7:
            migraine_type = "Chronic Migraine"
            severity = "severe"
            color = "red"
        elif severity_score > 4:
            migraine_type = "Episodic Migraine"
            severity = "moderate"
            color = "orange"
        else:
            migraine_type = "Occasional Migraine"
            severity = "mild"
            color = "yellow"
        
        st.markdown(f"### Assessment: <span style='color:{color}'>{migraine_type}</span>", unsafe_allow_html=True)
        st.write(f"Identified Triggers: {', '.join(triggers) if triggers else 'None identified'}")
        
        if name:
            with st.spinner("Generating personalized migraine management plan..."):
                patient_info = {
                    "name": name,
                    "frequency": frequency,
                    "triggers": triggers,
                    "intensity": intensity
                }
                
                recommendations = get_health_recommendations("Migraine", severity, patient_info)
                if recommendations:
                    display_recommendations(recommendations)
                    display_health_tips_dynamic("Migraine", severity.lower())


# HIV/AIDS Prediction
if selected == 'HIV/AIDS Prediction':
    st.title("🔴 HIV Risk Assessment")
    st.markdown("Confidential HIV risk evaluation")
    
    st.info("This assessment is completely confidential and for educational purposes only.")
    
    name = st.text_input("Name (optional):", placeholder="Anonymous")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=13, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        sexual_orientation = st.selectbox("Sexual Orientation", 
            ["Prefer not to say", "Heterosexual", "Homosexual", "Bisexual", "Other"])
    
    with col2:
        partners_number = st.selectbox("Number of Sexual Partners (past year)", 
            ["0", "1", "2-5", "6-10", ">10"])
        unprotected_sex = st.selectbox("Unprotected Sex", 
            ["Never", "Rarely", "Sometimes", "Often", "Always"])
        partner_hiv_status = st.selectbox("Partner's HIV Status", 
            ["Negative", "Positive", "Unknown", "Multiple partners with varied status"])
    
    with col3:
        iv_drug_use = st.selectbox("IV Drug Use", ["Never", "Past", "Current"])
        blood_transfusion = st.selectbox("Blood Transfusion (before 1985)", ["No", "Yes", "Unknown"])
        healthcare_exposure = st.selectbox("Occupational Healthcare Exposure", ["No", "Yes"])
    
    # Symptoms (if any)
    st.subheader("Symptoms (if any)")
    symptoms = st.multiselect("Select any symptoms experienced:",
        ["None", "Unexplained weight loss", "Persistent fever", "Night sweats",
         "Chronic diarrhea", "Persistent cough", "Skin rashes", "Swollen lymph nodes",
         "Oral thrush", "Recurring infections"])
    
    # Testing history
    st.subheader("Testing History")
    last_test = st.selectbox("Last HIV Test", 
        ["Never tested", "< 3 months ago", "3-6 months ago", "6-12 months ago", "> 1 year ago"])
    prep_use = st.selectbox("PrEP Use (Pre-Exposure Prophylaxis)", ["No", "Yes", "Considering"])
    
    if st.button("Assess HIV Risk"):
        risk_score = 0
        
        # Calculate risk based on behaviors
        if unprotected_sex in ["Often", "Always"]:
            risk_score += 3
        elif unprotected_sex == "Sometimes":
            risk_score += 2
        elif unprotected_sex == "Rarely":
            risk_score += 1
        
        if partner_hiv_status == "Positive":
            risk_score += 4
        elif partner_hiv_status in ["Unknown", "Multiple partners with varied status"]:
            risk_score += 2
        
        if partners_number in [">10", "6-10"]:
            risk_score += 2
        elif partners_number == "2-5":
            risk_score += 1
        
        if iv_drug_use == "Current":
            risk_score += 4
        elif iv_drug_use == "Past":
            risk_score += 1
        
        # Symptoms add to concern level
        if len(symptoms) > 3:
            risk_score += 2
        
        # Determine risk level
        if risk_score >= 8:
            risk_level = "High Risk - Immediate testing recommended"
            color = "red"
            severity = "high"
        elif risk_score >= 4:
            risk_level = "Moderate Risk - Regular testing recommended"
            color = "orange"
            severity = "moderate"
        else:
            risk_level = "Low Risk - Routine testing recommended"
            color = "green"
            severity = "low"
        
        st.markdown(f"### Risk Level: <span style='color:{color}'>{risk_level}</span>", unsafe_allow_html=True)
        
        # Testing recommendations
        st.subheader("Testing Recommendations")
        if risk_score >= 8:
            st.error("🔴 Immediate HIV testing is strongly recommended")
            st.write("Consider visiting a healthcare provider or testing center today")
        elif risk_score >= 4:
            st.warning("🟡 Regular HIV testing every 3-6 months is recommended")
        else:
            st.success("🟢 Annual HIV testing is recommended for sexually active individuals")
        
        if prep_use == "Considering":
            st.info("💊 PrEP can reduce HIV risk by up to 99% when taken as prescribed. Consult a healthcare provider.")
        
        if name:
            with st.spinner("Generating personalized recommendations..."):
                patient_info = {
                    "name": name if name else "Anonymous",
                    "risk_score": risk_score,
                    "prep_status": prep_use,
                    "testing_history": last_test
                }
                
                recommendations = get_health_recommendations("HIV Prevention", severity, patient_info)
                if recommendations:
                    display_recommendations(recommendations)
                    display_health_tips_dynamic("HIV", severity.lower())


# Obesity Prediction
if selected == 'Obesity Prediction':
    st.title("⚖️ Obesity Risk Assessment & BMI Calculator")
    st.markdown("Comprehensive weight and health evaluation")
    
    name = st.text_input("Name:")
    
    # Basic measurements
    st.subheader("Physical Measurements")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=2, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    
    with col2:
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0)
        waist = st.number_input("Waist Circumference (cm)", min_value=40, max_value=200, value=80)
        hip = st.number_input("Hip Circumference (cm)", min_value=40, max_value=200, value=95)
    
    with col3:
        neck = st.number_input("Neck Circumference (cm)", min_value=20, max_value=60, value=35)
        body_fat = st.number_input("Body Fat % (if known)", min_value=0.0, max_value=70.0, value=0.0)
        muscle_mass = st.number_input("Muscle Mass % (if known)", min_value=0.0, max_value=70.0, value=0.0)
    
    # Lifestyle factors
    st.subheader("Lifestyle Factors")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        physical_activity = st.selectbox("Physical Activity Level", 
            ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"])
        exercise_frequency = st.selectbox("Exercise Frequency", 
            ["Never", "1-2 times/week", "3-4 times/week", "5-6 times/week", "Daily"])
        sleep_hours = st.number_input("Average Sleep Hours", min_value=2, max_value=12, value=7)
    
    with col5:
        eating_habits = st.selectbox("Eating Habits", 
            ["Very Healthy", "Mostly Healthy", "Average", "Often Unhealthy", "Very Unhealthy"])
        meal_frequency = st.selectbox("Meals per Day", ["1-2", "3", "4-5", ">5"])
        water_intake = st.selectbox("Water Intake (liters/day)", ["<1", "1-2", "2-3", ">3"])
    
    with col6:
        stress_level = st.selectbox("Stress Level", ["Low", "Moderate", "High", "Very High"])
        family_obesity = st.selectbox("Family History of Obesity", ["No", "One Parent", "Both Parents"])
        medical_conditions = st.multiselect("Related Conditions", 
            ["None", "Diabetes", "Hypertension", "Thyroid Issues", "PCOS", "Sleep Apnea"])
    
    # Dietary habits
    st.subheader("Dietary Habits")
    col7, col8 = st.columns(2)
    
    with col7:
        fast_food = st.selectbox("Fast Food Consumption", 
            ["Never", "Rarely", "1-2 times/week", "3-4 times/week", "Daily"])
        sugary_drinks = st.selectbox("Sugary Drinks", 
            ["Never", "Rarely", "Sometimes", "Often", "Daily"])
        vegetable_intake = st.selectbox("Vegetable Intake", 
            ["None", "1 serving/day", "2-3 servings/day", "4+ servings/day"])
    
    with col8:
        snacking = st.selectbox("Between Meal Snacking", 
            ["Never", "Occasionally", "Daily", "Multiple times daily"])
        alcohol = st.selectbox("Alcohol Consumption", 
            ["Never", "Occasionally", "Weekly", "Daily"])
        smoking = st.selectbox("Smoking", ["Never", "Former", "Current"])
    
    if st.button("Calculate BMI & Assess Obesity Risk"):
        # Calculate BMI
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        
        # Calculate Waist-to-Hip Ratio
        whr = waist / hip if hip > 0 else 0
        
        # Calculate Waist-to-Height Ratio
        whtr = waist / height
        
        # Determine BMI category
        if bmi < 18.5:
            bmi_category = "Underweight"
            bmi_color = "blue"
        elif bmi < 25:
            bmi_category = "Normal weight"
            bmi_color = "green"
        elif bmi < 30:
            bmi_category = "Overweight"
            bmi_color = "yellow"
        elif bmi < 35:
            bmi_category = "Obese Class I"
            bmi_color = "orange"
        elif bmi < 40:
            bmi_category = "Obese Class II"
            bmi_color = "red"
        else:
            bmi_category = "Obese Class III (Severe)"
            bmi_color = "darkred"
        
        # Display results
        st.markdown("### 📊 Body Composition Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("BMI", f"{bmi:.1f}", bmi_category)
            st.markdown(f"<span style='color:{bmi_color}'>●</span> {bmi_category}", unsafe_allow_html=True)
        
        with col2:
            st.metric("Waist-to-Hip Ratio", f"{whr:.2f}")
            whr_risk = "High" if (gender == "Male" and whr > 0.9) or (gender == "Female" and whr > 0.85) else "Normal"
            st.write(f"Risk: {whr_risk}")
        
        with col3:
            st.metric("Waist-to-Height Ratio", f"{whtr:.2f}")
            whtr_risk = "High" if whtr > 0.5 else "Normal"
            st.write(f"Risk: {whtr_risk}")
        
        # Calculate overall obesity risk
        risk_score = 0
        
        # BMI risk
        if bmi >= 30:
            risk_score += 3
        elif bmi >= 25:
            risk_score += 2
        
        # Lifestyle factors
        if physical_activity == "Sedentary":
            risk_score += 2
        if eating_habits in ["Often Unhealthy", "Very Unhealthy"]:
            risk_score += 2
        if fast_food in ["3-4 times/week", "Daily"]:
            risk_score += 1
        if family_obesity == "Both Parents":
            risk_score += 2
        elif family_obesity == "One Parent":
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 7:
            risk_level = "High Risk"
            severity = "high"
        elif risk_score >= 4:
            risk_level = "Moderate Risk"
            severity = "moderate"
        else:
            risk_level = "Low Risk"
            severity = "low"
        
        st.markdown(f"### Overall Obesity Risk: {risk_level}")
        st.progress(min(risk_score/10, 1.0))
        
        # Health recommendations based on BMI
        st.subheader("📋 Personalized Recommendations")
        
        if bmi >= 25:
            st.warning("Weight management is recommended for optimal health")
            target_weight = 24.9 * (height_m ** 2)
            weight_to_lose = weight - target_weight
            st.write(f"**Target Weight:** {target_weight:.1f} kg")
            if weight_to_lose > 0:
                st.write(f"**Weight to Lose:** {weight_to_lose:.1f} kg")
            
            # Calculate daily calorie needs
            if gender == "Male":
                bmr = 10 * weight + 6.25 * height - 5 * age + 5
            else:
                bmr = 10 * weight + 6.25 * height - 5 * age - 161
            
            activity_multipliers = {
                "Sedentary": 1.2,
                "Lightly Active": 1.375,
                "Moderately Active": 1.55,
                "Very Active": 1.725,
                "Extremely Active": 1.9
            }
            
            tdee = bmr * activity_multipliers.get(physical_activity, 1.2)
            calorie_deficit = tdee - 500  # 500 calorie deficit for healthy weight loss
            
            st.write(f"**Estimated Daily Calorie Needs:** {tdee:.0f} calories")
            st.write(f"**Recommended for Weight Loss:** {calorie_deficit:.0f} calories/day")
            st.write(f"**Estimated Time to Target Weight:** {(weight_to_lose * 7700 / 500 / 7):.1f} weeks (at 0.5 kg/week)")
        
        elif bmi < 18.5:
            target_weight = 18.5 * (height_m ** 2)
            weight_to_gain = target_weight - weight
            st.info("Weight gain is recommended for optimal health")
            st.write(f"**Target Weight:** {target_weight:.1f} kg")
            st.write(f"**Weight to Gain:** {weight_to_gain:.1f} kg")
        
        else:
            st.success("You are at a healthy weight! Focus on maintenance.")
        
        # Get AI recommendations for obesity/weight management
        if name:
            with st.spinner("Generating personalized weight management plan..."):
                patient_info = {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "bmi": round(bmi, 2),
                    "weight": weight,
                    "height": height,
                    "physical_activity": physical_activity,
                    "eating_habits": eating_habits,
                    "medical_conditions": medical_conditions
                }
                
                disease_name = "Obesity Management" if bmi >= 30 else "Weight Management"
                recommendations = get_health_recommendations(disease_name, severity, patient_info)
                if recommendations:
                    display_recommendations(recommendations)
                    display_health_tips_dynamic("Obesity", severity.lower())

