"""LLM helper utilities for NVIDIA-hosted OpenAI-compatible models."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import streamlit as st
from openai import OpenAI

LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
_WARNING_FLAG = "_nvidia_api_warning_shown"
_CLIENT: Optional[OpenAI] = None


def initialize_llm_client() -> Optional[OpenAI]:
    """Create (or reuse) the NVIDIA-backed OpenAI client."""

    global _CLIENT  # noqa: PLW0603 - module cached for Streamlit reruns

    if _CLIENT is not None:
        return _CLIENT

    api_key = st.secrets.get("NVIDIA_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        if not st.session_state.get(_WARNING_FLAG, False):
            st.warning(
                "NVIDIA API key not configured. AI-generated recommendations and tips are currently disabled."
            )
            st.session_state[_WARNING_FLAG] = True
        return None

    _CLIENT = OpenAI(base_url=LLM_BASE_URL, api_key=api_key)
    return _CLIENT


def _clean_response(raw_response: str) -> str:
    cleaned = raw_response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def get_health_recommendations(
    client: Optional[OpenAI],
    disease_name: str,
    severity: str = "moderate",
    patient_info: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Request structured health recommendations for a disease."""

    if client is None:
        st.info("Configure the NVIDIA API key to enable AI-generated care plans.")
        return None

    payload = json.dumps(patient_info or {})
    prompt = f"""You are a medical AI assistant. Based on the following information, provide structured health recommendations.

Disease: {disease_name}
Severity: {severity}
Patient Information: {payload}

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
                {
                    "role": "system",
                    "content": "You are a medical AI assistant. You MUST return only valid JSON with no additional text.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            top_p=0.7,
            max_tokens=2048,
            stream=True,
        )

        full_response = ""
        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                full_response += delta

        cleaned = _clean_response(full_response)
        return json.loads(cleaned)

    except json.JSONDecodeError as exc:
        st.error(f"Failed to parse recommendations. Error: {exc}")
    except Exception as exc:  # noqa: BLE001 - surface runtime issues
        st.error(f"Error getting recommendations: {exc}")

    return None


def get_health_tips_from_llm(
    client: Optional[OpenAI],
    disease_name: str,
    severity: str = "moderate",
) -> Optional[Dict[str, Any]]:
    """Fetch disease-specific health tips."""

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
            model="writer/palmyra-med-70b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical AI assistant. You MUST return only valid JSON with no additional text.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            top_p=0.7,
            max_tokens=1536,
            stream=True,
        )

        full_response = ""
        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                full_response += delta

        cleaned = _clean_response(full_response)
        return json.loads(cleaned)

    except json.JSONDecodeError as exc:
        st.error(f"Failed to parse health tips. Error: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error getting health tips from LLM: {exc}")

    return None


def get_general_health_tips_from_llm(client: Optional[OpenAI]) -> Optional[Dict[str, Any]]:
    """Fetch generic wellness guidance."""

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
            model="writer/palmyra-med-70b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical AI assistant. You MUST return only valid JSON with no additional text.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            top_p=0.7,
            max_tokens=2048,
            stream=True,
        )

        full_response = ""
        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                full_response += delta

        cleaned = _clean_response(full_response)
        return json.loads(cleaned)

    except json.JSONDecodeError as exc:
        st.error(f"Failed to parse general health tips. Error: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error getting general health tips: {exc}")

    return None
