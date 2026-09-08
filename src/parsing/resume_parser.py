"""Module 1 — Resume parser.

Read an uploaded resume using loader.load_text(), send it to the LLM
with the resume parsing prompt, and return validated structured JSON
containing:

name, skills, experience, education, target_role.

Uses Gemini structured output.
"""

import sys
import os
import json
from pathlib import Path
from typing import TypedDict, List

import streamlit as st


# ============================================================
# 1. FIND PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. IMPORT REQUIRED LIBRARIES
# ============================================================

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src import config
from src.parsing.loader import load_text
from src.generate.prompts import RESUME_PARSE_PROMPT


# ============================================================
# 3. LOAD API KEY
# ============================================================

# Load local .env file first.
env_file = PROJECT_ROOT / ".env"

if env_file.exists():
    load_dotenv(env_file)


# Try local .env first.
api_key = os.getenv("GOOGLE_API_KEY")


# If .env does not contain the key,
# try Streamlit Cloud Secrets.
if not api_key:

    try:
        api_key = st.secrets["GOOGLE_API_KEY"]

    except Exception:
        api_key = None


# Stop with a clear error if no key exists.
if not api_key:

    raise ValueError(
        f"{config.API_KEY_ENV} not found. "
        "For local use, add GOOGLE_API_KEY to .env. "
        "For Streamlit Cloud, add GOOGLE_API_KEY to Secrets."
    )


# ============================================================
# 4. CONNECT TO GEMINI
# ============================================================

client = genai.Client(
    api_key=api_key
)


# Model comes from config.py
MODEL = config.CHAT_MODEL


# ============================================================
# 5. DEFINE RESUME STRUCTURE
# ============================================================

class Education(TypedDict):

    degree: str
    institute: str
    year: str
    score: str


class Experience(TypedDict):

    company: str
    role: str
    duration: str
    highlights: List[str]


class Resume(TypedDict):

    name: str
    skills: List[str]
    experience: List[Experience]
    education: List[Education]
    target_role: str


# ============================================================
# 6. VALIDATE RESUME JSON
# ============================================================

def validate_resume(data: dict) -> Resume:

    """Validate the structured resume returned by Gemini."""

    required_fields = {
        "name",
        "skills",
        "experience",
        "education",
        "target_role",
    }


    # --------------------------------------------------------
    # Check required top-level fields
    # --------------------------------------------------------

    missing_fields = (
        required_fields - set(data.keys())
    )


    if missing_fields:

        raise ValueError(
            f"Missing required resume fields: {missing_fields}"
        )


    # --------------------------------------------------------
    # Check top-level types
    # --------------------------------------------------------

    if not isinstance(data["name"], str):

        raise ValueError(
            "name must be a string"
        )


    if not isinstance(data["skills"], list):

        raise ValueError(
            "skills must be a list"
        )


    if not isinstance(data["experience"], list):

        raise ValueError(
            "experience must be a list"
        )


    if not isinstance(data["education"], list):

        raise ValueError(
            "education must be a list"
        )


    if not isinstance(data["target_role"], str):

        raise ValueError(
            "target_role must be a string"
        )


    # ========================================================
    # Validate experience
    # ========================================================

    experience_fields = {
        "company",
        "role",
        "duration",
        "highlights",
    }


    for experience in data["experience"]:

        if not isinstance(experience, dict):

            raise ValueError(
                "Each experience item must be a dictionary"
            )


        missing = (
            experience_fields
            - set(experience.keys())
        )


        if missing:

            raise ValueError(
                f"Experience item missing fields: {missing}"
            )


        if not isinstance(
            experience["company"],
            str,
        ):

            raise ValueError(
                "experience.company must be a string"
            )


        if not isinstance(
            experience["role"],
            str,
        ):

            raise ValueError(
                "experience.role must be a string"
            )


        if not isinstance(
            experience["duration"],
            str,
        ):

            raise ValueError(
                "experience.duration must be a string"
            )


        if not isinstance(
            experience["highlights"],
            list,
        ):

            raise ValueError(
                "experience.highlights must be a list"
            )


    # ========================================================
    # Validate education
    # ========================================================

    education_fields = {
        "degree",
        "institute",
        "year",
        "score",
    }


    for education in data["education"]:

        if not isinstance(education, dict):

            raise ValueError(
                "Each education item must be a dictionary"
            )


        missing = (
            education_fields
            - set(education.keys())
        )


        if missing:

            raise ValueError(
                f"Education item missing fields: {missing}"
            )


        if not isinstance(
            education["degree"],
            str,
        ):

            raise ValueError(
                "education.degree must be a string"
            )


        if not isinstance(
            education["institute"],
            str,
        ):

            raise ValueError(
                "education.institute must be a string"
            )


        if not isinstance(
            education["year"],
            str,
        ):

            raise ValueError(
                "education.year must be a string"
            )


        if not isinstance(
            education["score"],
            str,
        ):

            raise ValueError(
                "education.score must be a string"
            )


    return data


# ============================================================
# 7. PARSE RESUME TEXT
# ============================================================

def parse_resume(text: str) -> Resume:

    """Parse resume text using Gemini structured output."""

    if not isinstance(text, str):

        raise TypeError(
            "Resume text must be a string."
        )


    text = text.strip()


    if not text:

        raise ValueError(
            "Resume text cannot be empty."
        )


    # --------------------------------------------------------
    # Get prompt
    # --------------------------------------------------------

    prompt = RESUME_PARSE_PROMPT + f"""

RESUME:

{text}

"""


    # --------------------------------------------------------
    # Gemini structured output
    # --------------------------------------------------------

    response = client.models.generate_content(

        model=MODEL,

        contents=prompt,

        config=types.GenerateContentConfig(

            response_mime_type="application/json",

            response_schema=Resume,

            temperature=0.2,

            max_output_tokens=2048,
        ),
    )


    # --------------------------------------------------------
    # Check response
    # --------------------------------------------------------

    if not response.text:

        raise ValueError(
            "Gemini returned an empty response."
        )


    # --------------------------------------------------------
    # Convert JSON string to Python dictionary
    # --------------------------------------------------------

    try:

        parsed_data = json.loads(
            response.text
        )

    except json.JSONDecodeError as e:

        raise ValueError(
            f"Gemini returned invalid JSON: {e}"
        )


    # --------------------------------------------------------
    # Validate before returning
    # --------------------------------------------------------

    return validate_resume(
        parsed_data
    )


# ============================================================
# 8. PARSE RESUME FILE
# ============================================================

def parse_resume_file(path) -> Resume:

    """Load a resume file using loader.py and parse it."""

    resume_text = load_text(
        path
    )

    return parse_resume(
        resume_text
    )


# ============================================================
# 9. TEST THE RESUME PARSER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "SMART HIRE - RESUME PARSER"
    )

    print("=" * 60)


    resume_folder = config.RESUMES_DIR


    if not resume_folder.exists():

        raise FileNotFoundError(
            f"Resume folder not found: {resume_folder}"
        )


    resume_files = [

        file

        for file in resume_folder.iterdir()

        if file.suffix.lower()
        in [
            ".pdf",
            ".docx",
            ".txt",
            ".md",
        ]
    ]


    print(
        f"\nResumes found: {len(resume_files)}"
    )


    for file in resume_files:

        print(
            "-",
            file.name
        )


    # ========================================================
    # Process all resumes
    # ========================================================

    for file in resume_files:

        print(
            "\n" + "=" * 60
        )

        print(
            "PROCESSING:",
            file.name
        )

        print(
            "=" * 60
        )


        try:

            parsed_data = parse_resume_file(
                file
            )


            print(
                "\nSTRUCTURED OUTPUT:"
            )


            print(

                json.dumps(

                    parsed_data,

                    indent=2,

                    ensure_ascii=False,
                )
            )


            print(
                "\nVALIDATION: SUCCESS"
            )


        except Exception as e:

            print(
                "\nERROR:"
            )

            print(e)


    print(
        "\n" + "=" * 60
    )

    print(
        "RESUME PARSER COMPLETED"
    )

    print(
        "=" * 60
    )