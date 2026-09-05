"""Module 3 — CV improvement generator.

Given the parsed resume and a target job, prompt the LLM (using the prompt from
prompts.py) to return specific suggestions: missing skills, weak bullet points, and
a rewritten summary. Prototype the prompt in a notebook, then move it here.
"""

"""Module 3 — CV improvement generator.

Given the parsed resume and a target job, prompt the LLM
using the prompt from prompts.py to return specific
suggestions: missing skills, weak bullet points, and
a rewritten summary.
"""

import sys
import os
import json
from pathlib import Path
from typing import TypedDict, List


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src import config
from src.generate.prompts import CV_SUGGESTIONS_PROMPT


# ============================================================
# LOAD API KEY
# ============================================================

load_dotenv(PROJECT_ROOT / ".env.example")

api_key = os.getenv(config.API_KEY_ENV)

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY not found in .env.example"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=api_key
)


# ============================================================
# MODEL
# ============================================================

MODEL = "gemini-3.5-flash-lite"


# ============================================================
# OUTPUT SCHEMA
# ============================================================

class CVSuggestion(TypedDict):
    missing_skills: List[str]
    weak_bullet_points: List[str]
    rewritten_summary: str


# ============================================================
# GENERATE CV SUGGESTIONS
# ============================================================

def generate_cv_suggestions(
    resume: dict,
    target_job: dict
) -> CVSuggestion:
    """
    Compare a parsed resume with a target job and generate
    CV improvement suggestions.

    Args:
        resume:
            Parsed resume dictionary.

        target_job:
            Target job dictionary.

    Returns:
        Dictionary containing:
        - missing_skills
        - weak_bullet_points
        - rewritten_summary
    """

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if not isinstance(resume, dict):
        raise TypeError(
            "resume must be a dictionary"
        )

    if not isinstance(target_job, dict):
        raise TypeError(
            "target_job must be a dictionary"
        )


    # --------------------------------------------------------
    # Convert inputs to JSON
    # --------------------------------------------------------

    resume_json = json.dumps(
        resume,
        indent=2,
        ensure_ascii=False
    )

    target_job_json = json.dumps(
        target_job,
        indent=2,
        ensure_ascii=False
    )


    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------
    # Use replace() instead of .format() because the prompt
    # may contain JSON braces { }.

    prompt = CV_SUGGESTIONS_PROMPT.replace(
        "{resume}",
        resume_json
    ).replace(
        "{target_job}",
        target_job_json
    )


    # --------------------------------------------------------
    # Call Gemini
    # --------------------------------------------------------

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CVSuggestion,
            temperature=0.2,
            max_output_tokens=2048
        )
    )


    # --------------------------------------------------------
    # Parse response
    # --------------------------------------------------------

    try:

        result = json.loads(response.text)

    except json.JSONDecodeError as e:

        raise ValueError(
            f"Gemini returned invalid JSON: {e}"
        )


    # --------------------------------------------------------
    # Validate fields
    # --------------------------------------------------------

    required_fields = {
        "missing_skills",
        "weak_bullet_points",
        "rewritten_summary"
    }

    missing_fields = (
        required_fields - set(result.keys())
    )

    if missing_fields:

        raise ValueError(
            f"Missing required fields: {missing_fields}"
        )


    # --------------------------------------------------------
    # Validate data types
    # --------------------------------------------------------

    if not isinstance(
        result["missing_skills"],
        list
    ):

        raise ValueError(
            "missing_skills must be a list"
        )


    if not isinstance(
        result["weak_bullet_points"],
        list
    ):

        raise ValueError(
            "weak_bullet_points must be a list"
        )


    if not isinstance(
        result["rewritten_summary"],
        str
    ):

        raise ValueError(
            "rewritten_summary must be a string"
        )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return result