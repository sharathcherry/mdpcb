"""Streamlit rendering helpers for health recommendations and tips."""

from __future__ import annotations

import datetime
from typing import Any, Dict, Optional

import streamlit as st

from new.services.llm import (
    get_general_health_tips_from_llm,
    get_health_tips_from_llm,
)


def generate_text_report(recommendations: Dict[str, Any]) -> str:
    """Generate a plain-text report summarizing an AI recommendation."""

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

    dietary = recommendations.get("dietary_plan", {})
    if dietary:
        report += "\nDaily Nutritional Targets:\n"
        report += f"- Calories: {dietary.get('daily_calories', 'Not specified')}\n"
        report += f"- Protein: {dietary.get('daily_protein', 'Not specified')}\n"
        report += f"- Carbohydrates: {dietary.get('daily_carbohydrates', 'Not specified')}\n"
        report += f"- Fats: {dietary.get('daily_fats', 'Not specified')}\n"
        report += f"- Fiber: {dietary.get('daily_fiber', 'Not specified')}\n"
        report += f"- Water: {dietary.get('hydration', 'Not specified')}\n"

        report += "\nFoods to Eat:\n"
        for food in dietary.get("foods_to_eat", []):
            report += f"- {food}\n"

        report += "\nFoods to Avoid:\n"
        for food in dietary.get("foods_to_avoid", []):
            report += f"- {food}\n"

    report += f"\n{'='*80}\nMEDICATIONS\n{'='*80}\n\n"

    medications = recommendations.get("medications", {})
    if medications:
        for med in medications.get("medication_details", []):
            report += f"\n{med.get('name', 'Medication')}:\n"
            report += f"  Dosage: {med.get('dosage', 'N/A')}\n"
            report += f"  Frequency: {med.get('frequency', 'N/A')}\n"
            report += f"  Duration: {med.get('duration', 'N/A')}\n"

    report += f"\n{'='*80}\nDOCTOR VISITATION\n{'='*80}\n\n"

    doctor = recommendations.get("doctor_visitation", {})
    if doctor:
        report += f"Urgency: {doctor.get('urgency', 'N/A')}\n"
        report += f"Specialist: {doctor.get('specialist_type', 'N/A')}\n"
        report += f"Follow-up: {doctor.get('followup_schedule', 'N/A')}\n"

    report += f"\n{'='*80}\nPRECAUTIONS\n{'='*80}\n\n"

    precautions = recommendations.get("precautions", {})
    if precautions:
        report += "Lifestyle Changes:\n"
        for change in precautions.get("lifestyle_changes", []):
            report += f"- {change}\n"

        report += "\nWarning Signs:\n"
        for sign in precautions.get("warning_signs", []):
            report += f"- {sign}\n"

    report += f"\n{'='*80}\nEXERCISE RECOMMENDATIONS\n{'='*80}\n\n"

    exercise = recommendations.get("exercise_recommendations", {})
    if exercise:
        report += f"Duration: {exercise.get('duration', 'N/A')}\n"
        report += f"Frequency: {exercise.get('frequency', 'N/A')}\n"
        report += f"Intensity: {exercise.get('intensity', 'N/A')}\n"
        report += "\nRecommended Exercises:\n"
        for ex in exercise.get("recommended_exercises", []):
            report += f"- {ex}\n"

    report += f"\n{'='*80}\n"
    report += "DISCLAIMER: This report is for informational purposes only.\n"
    report += "Please consult with qualified healthcare professionals for medical advice.\n"
    report += f"{'='*80}\n"

    return report


def display_recommendations(recommendations: Optional[Dict[str, Any]]) -> None:
    """Render the AI-generated recommendation bundle."""

    if not recommendations:
        st.warning("Unable to generate recommendations at this time.")
        return

    st.markdown("---")
    st.markdown("## 📋 Health Management Plan")
    if recommendations.get("name"):
        st.markdown(f"**Patient:** {recommendations.get('name')}")
    st.markdown(f"**Condition:** {recommendations.get('topic', 'Health Management')}")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🍽️ Diet Plan",
        "💊 Medications",
        "👨‍⚕️ Doctor Visit",
        "⚠️ Precautions",
        "🏃 Exercise",
    ])

    dietary = recommendations.get("dietary_plan", {})
    medications = recommendations.get("medications", {})
    doctor = recommendations.get("doctor_visitation", {})
    precautions = recommendations.get("precautions", {})
    exercise = recommendations.get("exercise_recommendations", {})

    with tab1:
        st.subheader("🍽️ Dietary Plan")
        if dietary:
            st.markdown("### 📊 Daily Nutritional Targets")
            nutrition_cols = st.columns(4)
            metrics = [
                ("Calories", dietary.get("daily_calories", "Not specified")),
                ("Protein", dietary.get("daily_protein", "Not specified")),
                ("Carbohydrates", dietary.get("daily_carbohydrates", "Not specified")),
                ("Healthy Fats", dietary.get("daily_fats", "Not specified")),
            ]
            for col, (label, value) in zip(nutrition_cols, metrics):
                with col:
                    st.metric(label, value)

            st.markdown("---")
            st.markdown("### 🔬 Key Nutritional Guidelines")
            nutrition_cols2 = st.columns(5)
            metrics2 = [
                ("Fiber", dietary.get("daily_fiber", "Not specified")),
                ("Sodium (max)", dietary.get("daily_sodium", "Not specified")),
                ("Added Sugar (max)", dietary.get("daily_sugar", "Not specified")),
                ("Cholesterol (max)", dietary.get("daily_cholesterol", "Not specified")),
                ("Water", dietary.get("hydration", "Not specified")),
            ]
            for col, (label, value) in zip(nutrition_cols2, metrics2):
                with col:
                    st.metric(label, value)

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ✅ Foods to Eat:")
                foods_to_eat = dietary.get("foods_to_eat", [])
                if foods_to_eat:
                    for food in foods_to_eat:
                        st.markdown(f"• {food}")
                else:
                    st.info("No specific recommendations")
            with col2:
                st.markdown("### ❌ Foods to Avoid:")
                foods_to_avoid = dietary.get("foods_to_avoid", [])
                if foods_to_avoid:
                    for food in foods_to_avoid:
                        st.markdown(f"• {food}")
                else:
                    st.info("No specific restrictions")

            st.markdown("---")
            st.markdown("### 📅 Sample Meal Plan")
            meal_plan = dietary.get("meal_plan", {})
            if meal_plan:
                meal_cols = st.columns(4)
                meals = [
                    ("🌅 Breakfast", meal_plan.get("breakfast", "")),
                    ("☀️ Lunch", meal_plan.get("lunch", "")),
                    ("🌆 Dinner", meal_plan.get("dinner", "")),
                    ("🍎 Snacks", meal_plan.get("snacks", "")),
                ]
                for col, (meal_name, meal_content) in zip(meal_cols, meals):
                    with col:
                        st.markdown(f"**{meal_name}**")
                        st.write(meal_content if meal_content else "Not specified")

            st.markdown("---")
            st.markdown("### 💊 Essential Vitamins & Minerals")
            vitamins = dietary.get("vitamins_minerals", {})
            if vitamins:
                vit_cols = st.columns(3)
                with vit_cols[0]:
                    st.markdown("**Key Vitamins:**")
                    vit_dict = vitamins.get("vitamins", {})
                    if vit_dict:
                        for vit, amount in vit_dict.items():
                            st.markdown(f"• {vit}: {amount}")
                    else:
                        st.write("Standard daily requirements")
                with vit_cols[1]:
                    st.markdown("**Key Minerals:**")
                    min_dict = vitamins.get("minerals", {})
                    if min_dict:
                        for mineral, amount in min_dict.items():
                            st.markdown(f"• {mineral}: {amount}")
                    else:
                        st.write("Standard daily requirements")
                with vit_cols[2]:
                    st.markdown("**Supplements (if needed):**")
                    supplements = vitamins.get("supplements", [])
                    if supplements:
                        for supp in supplements:
                            st.markdown(f"• {supp}")
                    else:
                        st.write("Consult your doctor")

            st.markdown("---")
            st.markdown("### ⏰ Meal Timing & Frequency")
            timing = dietary.get("meal_timing", {})
            if timing:
                st.info(
                    f"**Recommended eating schedule:** {timing.get('schedule', 'Eat regular meals every 3-4 hours')}"
                )
                st.write(
                    f"**Best practices:** {timing.get('tips', 'Avoid eating 2-3 hours before bedtime')}"
                )
            else:
                st.info("Eat balanced meals at regular intervals throughout the day")

            st.markdown("---")
            st.markdown("### 🍛 Portion Control Guide")
            portions = dietary.get("portion_sizes", {})
            if portions:
                portion_cols = st.columns(2)
                with portion_cols[0]:
                    st.markdown("**Recommended Portions:**")
                    for food_group, portion in portions.items():
                        st.markdown(f"• {food_group}: {portion}")
                with portion_cols[1]:
                    st.info(
                        "**Hand-based portion guide:**\n\n"
                        "• Palm = Protein serving\n"
                        "• Fist = Vegetable serving\n"
                        "• Cupped hand = Carb serving\n"
                        "• Thumb = Fat serving"
                    )
            else:
                st.info("Follow standard portion guidelines based on your age, gender, and activity level")

    with tab2:
        st.subheader("💊 Medications")
        if medications:
            prescription = medications.get("prescription_required", [])
            if prescription:
                st.markdown("### 🏥 Prescription Required")
                for med in prescription:
                    st.markdown(f"• {med}")
            otc = medications.get("over_the_counter", [])
            if otc:
                st.markdown("### 🛒 Over-the-Counter Options")
                for med in otc:
                    st.markdown(f"• {med}")
            st.markdown("---")
            st.markdown("### 📝 Medication Details")
            med_details = medications.get("medication_details", [])
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
                            generic_alt = med.get("generic_alternatives", [])
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
        if doctor:
            urgency = doctor.get("urgency", "routine")
            urgency_colors = {
                "immediate": ("🔴", "red", "IMMEDIATE ATTENTION REQUIRED"),
                "within 24 hours": ("🟠", "orange", "URGENT - Within 24 Hours"),
                "within a week": ("🟡", "gold", "Schedule Within a Week"),
                "routine": ("🟢", "green", "Routine Check-up"),
            }
            icon, color, message = urgency_colors.get(urgency.lower(), ("🔵", "blue", urgency))
            st.markdown("### Urgency Level")
            st.markdown(
                f"<h3 style='color:{color}'>{icon} {message}</h3>",
                unsafe_allow_html=True,
            )
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🏥 Specialist Type")
                specialist = doctor.get("specialist_type", "General Practitioner")
                st.info(specialist)
                st.markdown("### 📅 Follow-up Schedule")
                followup = doctor.get("followup_schedule", doctor.get("follow_up_schedule", "As needed"))
                st.write(followup)
            with col2:
                st.markdown("### 🔬 Recommended Tests")
                tests = doctor.get("tests_recommended", [])
                if tests:
                    for test in tests:
                        st.markdown(f"• {test}")
                else:
                    st.write("To be determined by physician")
        else:
            st.info("Consult with your healthcare provider for personalized medical guidance")

    with tab4:
        st.subheader("⚠️ Precautions")
        if precautions:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ✅ Lifestyle Changes")
                lifestyle = precautions.get("lifestyle_changes", [])
                if lifestyle:
                    for change in lifestyle:
                        st.markdown(f"• {change}")
                else:
                    st.info("Maintain healthy lifestyle habits")
                st.markdown("### 🚫 Activities to Avoid")
                avoid = precautions.get("activities_to_avoid", [])
                if avoid:
                    for activity in avoid:
                        st.markdown(f"• {activity}")
                else:
                    st.info("No specific restrictions")
            with col2:
                st.markdown("### ⚠️ Warning Signs")
                warnings = precautions.get("warning_signs", [])
                if warnings:
                    for sign in warnings:
                        st.warning(f"• {sign}")
                else:
                    st.info("Monitor general health")
            st.markdown("---")
            emergency = precautions.get("emergency_symptoms", [])
            if emergency:
                st.markdown("### 🆘 Emergency Symptoms (Seek Immediate Help)")
                st.error("If you experience any of these symptoms, call emergency services immediately:")
                for symptom in emergency:
                    st.markdown(f"• {symptom}")
        else:
            st.info("Follow general health precautions and consult your doctor")

    with tab5:
        st.subheader("🏃 Exercise Recommendations")
        if exercise:
            col1, col2, col3 = st.columns(3)
            metrics = [
                ("Duration", exercise.get("duration", "N/A")),
                ("Frequency", exercise.get("frequency", "N/A")),
                ("Intensity", exercise.get("intensity", "N/A")),
            ]
            for col, (label, value) in zip((col1, col2, col3), metrics):
                with col:
                    st.metric(label, value)
            st.markdown("---")
            st.markdown("### 💪 Recommended Exercises")
            exercises = exercise.get("recommended_exercises", [])
            if exercises:
                for idx, item in enumerate(exercises, 1):
                    st.markdown(f"{idx}. {item}")
            else:
                st.info("Consult a fitness professional for personalized exercise plan")
            st.markdown("---")
            st.info(
                "⚠️ Always consult your doctor before starting a new exercise program, especially if you have existing health conditions."
            )
        else:
            st.info("Regular physical activity is important. Consult your doctor for personalized exercise recommendations.")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        patient_name = recommendations.get("name", "patient")
        topic = recommendations.get("topic", "health")
        download_key = f"download_btn_{patient_name.replace(' ', '_')}_{hash(str(topic))}"
        if st.button("📥 Download Full Report", use_container_width=True, key=download_key):
            report_text = generate_text_report(recommendations)
            st.download_button(
                label="📄 Download Text Report",
                data=report_text,
                file_name=(
                    f"health_report_{patient_name.replace(' ', '_')}"
                    f"_{datetime.datetime.now().strftime('%Y%m%d')}.txt"
                ),
                mime="text/plain",
                key=f"download_actual_{download_key}",
            )


def display_health_tips_dynamic(
    client,
    disease_name: Optional[str] = None,
    severity: Optional[str] = None,
) -> None:
    """Render disease-specific or general health tips using the LLM service."""

    st.markdown("---")
    st.markdown("## 💡 Health Tips & Daily Guidance")

    if disease_name:
        tip_tabs = st.tabs(["🎯 Disease-Specific Tips", "📋 General Health Tips", "⚠️ Warning Signs"])
        tips = None
        with tip_tabs[0]:
            st.subheader(f"Personalized Tips for {disease_name}")
            with st.spinner("Generating personalized health tips from AI..."):
                tips = get_health_tips_from_llm(client, disease_name, severity or "moderate")
            if tips:
                st.markdown("### 📅 Daily Management")
                for idx, tip in enumerate(tips.get("daily_management_tips", []), 1):
                    st.info(f"**Tip {idx}:** {tip}")
                st.markdown("---")
                st.markdown("### ✅ Do's and ❌ Don'ts")
                do_dont = tips.get("do_and_dont", {})
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### ✅ Things to DO")
                    for item in do_dont.get("do", []):
                        st.success(f"✓ {item}")
                with col2:
                    st.markdown("#### ❌ Things to AVOID")
                    for item in do_dont.get("dont", []):
                        st.error(f"✗ {item}")
                st.markdown("---")
                st.markdown("### 🛡️ Prevention & Long-term Care")
                for tip in tips.get("prevention_tips", []):
                    st.success(f"• {tip}")
                st.markdown("---")
                st.markdown("### 🔄 Lifestyle Modifications")
                lifestyle = tips.get("lifestyle_modifications", [])
                cols = st.columns(2)
                for idx, mod in enumerate(lifestyle):
                    with cols[idx % 2]:
                        st.info(f"• {mod}")
                st.markdown("---")
                st.markdown("### ⚡ Quick Daily Reminders")
                for tip in tips.get("quick_reminders", []):
                    st.markdown(f"⚡ {tip}")
            else:
                st.warning("Unable to generate personalized tips at this time. Please ensure API is configured correctly.")
                st.info(
                    """
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
                    """
                )

        with tip_tabs[1]:
            st.subheader("General Health & Wellness")
            with st.spinner("Loading general health tips from AI..."):
                general_tips = get_general_health_tips_from_llm(client)
            if general_tips:
                categories = {
                    "nutrition": "🍎 Nutrition & Diet",
                    "physical_activity": "🏃 Physical Activity",
                    "sleep_rest": "😴 Sleep & Rest",
                    "mental_health": "🧘 Mental Health",
                    "preventive_care": "💊 Preventive Care",
                    "lifestyle_habits": "🌟 Lifestyle Habits",
                    "hydration": "💧 Hydration",
                    "immune_health": "🛡️ Immune Health",
                }
                for key, title in categories.items():
                    with st.expander(title):
                        for tip in general_tips.get(key, []):
                            st.markdown(f"• {tip}")
            else:
                st.warning("Unable to load general health tips. Using offline recommendations.")
                with st.expander("🍎 Nutrition & Diet"):
                    st.markdown(
                        """
                        - Eat a variety of colorful fruits and vegetables daily
                        - Choose whole grains over refined grains
                        - Include lean proteins in your diet
                        - Limit processed foods and added sugars
                        - Practice portion control
                        - Read nutrition labels carefully
                        - Plan meals ahead to make healthier choices
                        - Eat mindfully without distractions
                        """
                    )
                with st.expander("🏃 Physical Activity"):
                    st.markdown(
                        """
                        - Aim for 150 minutes of moderate activity weekly
                        - Include both cardio and strength training
                        - Start slowly and gradually increase intensity
                        - Find activities you enjoy for sustainability
                        - Take breaks from sitting every hour
                        - Stretch regularly to maintain flexibility
                        - Exercise with friends for motivation
                        - Listen to your body and rest when needed
                        """
                    )

        with tip_tabs[2]:
            st.subheader("⚠️ Warning Signs to Monitor")
            if tips:
                for sign in tips.get("warning_signs", []):
                    st.warning(f"• {sign}")
            else:
                st.warning(
                    """
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
                    """
                )
        return

    st.info("Select a specific health condition for personalized tips.")
    with st.spinner("Loading general wellness tips..."):
        general_tips = get_general_health_tips_from_llm(client)
    if general_tips:
        categories = {
            "nutrition": "🍎 Nutrition & Diet",
            "physical_activity": "🏃 Physical Activity",
            "sleep_rest": "😴 Sleep & Rest",
            "mental_health": "🧘 Mental Health",
            "preventive_care": "💊 Preventive Care",
            "lifestyle_habits": "🌟 Lifestyle Habits",
            "hydration": "💧 Hydration",
            "immune_health": "🛡️ Immune Health",
        }
        for key, title in categories.items():
            with st.expander(title):
                for tip in general_tips.get(key, []):
                    st.markdown(f"• {tip}")
