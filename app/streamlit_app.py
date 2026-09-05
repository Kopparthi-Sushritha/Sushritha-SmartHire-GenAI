"""SmartHire GenAI portal (Streamlit). Run: streamlit run app/streamlit_app.py

Build this LAST. It only wires together pieces that already work in src/:
    upload CV -> parsed profile -> matched jobs -> CV suggestions -> mentor chat.
Cache the FAISS index / chain with @st.cache_resource so it is not rebuilt on every
interaction, and read the API key from st.secrets (not the .env file) once deployed.
"""
import sys
import tempfile
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.parsing.loader import load_text

from src.parsing.resume_parser import parse_resume

from src.safety.guardrails import (
    check_input,
    is_resume_specific_question,
)

from src.search.job_search import search_jobs

from src.generate.cv_suggestions import (
    generate_cv_suggestions,
)

# IMPORTANT:
# Use ask_mentor because this is the public function
# provided by your actual RAG chain.
from src.mentor.rag_chain import ask_mentor


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartHire GenAI",
    page_icon="💼",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""


if "resume_profile" not in st.session_state:
    st.session_state.resume_profile = None


if "job_matches" not in st.session_state:
    st.session_state.job_matches = []


if "cv_suggestions" not in st.session_state:
    st.session_state.cv_suggestions = None


if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# TITLE
# ============================================================

st.title("💼 SmartHire GenAI")

st.write(
    "Resume Matching & AI Career Mentor"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Resume Analysis")


    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf", "docx", "txt"],
    )


    analyze_button = st.button(
        "🔍 Analyze Resume",
        use_container_width=True,
    )


# ============================================================
# RESUME ANALYSIS
# ============================================================

if analyze_button:

    if uploaded_file is None:

        st.sidebar.error(
            "Please upload a resume first."
        )

    else:

        try:

            # ------------------------------------------------
            # Save uploaded file temporarily
            # ------------------------------------------------

            suffix = Path(
                uploaded_file.name
            ).suffix


            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp_file.name


            # ------------------------------------------------
            # Read resume
            # ------------------------------------------------

            with st.spinner(
                "Reading your resume..."
            ):

                resume_text = load_text(
                    temp_path
                )


            if not resume_text.strip():

                st.error(
                    "Could not extract text from the resume."
                )

            else:

                # ------------------------------------------------
                # Parse resume
                # ------------------------------------------------

                with st.spinner(
                    "Analyzing your resume..."
                ):

                    resume_profile = parse_resume(
                        resume_text
                    )


                # ------------------------------------------------
                # Search matching jobs
                # ------------------------------------------------

                with st.spinner(
                    "Finding matching jobs..."
                ):

                    job_matches = search_jobs(
                        resume_text,
                        top_n=5,
                    )


                # ------------------------------------------------
                # Generate CV suggestions
                # ------------------------------------------------

                with st.spinner(
                    "Generating CV suggestions..."
                ):

                    if job_matches:

                        target_job = job_matches[0]

                        cv_suggestions = (
                            generate_cv_suggestions(
                                resume_profile,
                                target_job,
                            )
                        )

                    else:

                        cv_suggestions = None


                # ------------------------------------------------
                # Save results
                # ------------------------------------------------

                st.session_state.resume_text = (
                    resume_text
                )


                st.session_state.resume_profile = (
                    resume_profile
                )


                st.session_state.job_matches = (
                    job_matches
                )


                st.session_state.cv_suggestions = (
                    cv_suggestions
                )


                # ------------------------------------------------
                # Clear previous chat
                # ------------------------------------------------

                st.session_state.messages = []


                st.success(
                    "Resume analyzed successfully!"
                )


        except Exception as e:

            st.error(
                f"Error while analyzing resume: {e}"
            )


# ============================================================
# MAIN CONTENT
# ============================================================

if not st.session_state.resume_profile:

    st.info(
        "👈 Upload your resume from the sidebar "
        "and click **Analyze Resume** to get started."
    )


else:

    # ========================================================
    # RESUME PROFILE
    # ========================================================

    st.header("👤 Resume Profile")


    profile = (
        st.session_state.resume_profile
    )


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # NAME + TARGET ROLE
    # --------------------------------------------------------

    with col1:

        st.subheader("Name")


        st.write(
            profile.get(
                "name",
                "Not available",
            )
        )


        st.subheader("Target Role")


        st.write(
            profile.get(
                "target_role",
                "Not available",
            )
        )


    # --------------------------------------------------------
    # SKILLS
    # --------------------------------------------------------

    with col2:

        st.subheader("Skills")


        skills = profile.get(
            "skills",
            [],
        )


        if skills:

            st.write(
                ", ".join(skills)
            )

        else:

            st.write(
                "No skills found."
            )


    # ========================================================
    # EXPERIENCE
    # ========================================================

    st.subheader("💼 Experience")


    experience = profile.get(
        "experience",
        [],
    )


    if experience:

        for exp in experience:

            company = exp.get(
                "company",
                "Unknown company",
            )


            role = exp.get(
                "role",
                "Unknown role",
            )


            duration = exp.get(
                "duration",
                "",
            )


            st.markdown(
                f"**{role} — {company}**"
            )


            if duration:

                st.write(
                    f"Duration: {duration}"
                )


            highlights = exp.get(
                "highlights",
                [],
            )


            for highlight in highlights:

                st.write(
                    f"• {highlight}"
                )


    else:

        st.write(
            "No experience information found."
        )


    # ========================================================
    # EDUCATION
    # ========================================================

    st.subheader("🎓 Education")


    education = profile.get(
        "education",
        [],
    )


    if education:

        for edu in education:

            degree = edu.get(
                "degree",
                "Unknown degree",
            )


            institute = edu.get(
                "institute",
                "Unknown institute",
            )


            year = edu.get(
                "year",
                "",
            )


            st.markdown(
                f"**{degree}** — {institute}"
            )


            if year:

                st.write(
                    f"Year: {year}"
                )


    else:

        st.write(
            "No education information found."
        )


    st.divider()


    # ========================================================
    # MATCHING JOBS
    # ========================================================

    st.header("🎯 Matching Jobs")


    job_matches = (
        st.session_state.job_matches
    )


    if job_matches:

        for job in job_matches:

            rank = job.get(
                "rank",
                "",
            )


            title = job.get(
                "jobtitle",
                "Job",
            )


            score = job.get(
                "similarity_score",
                0,
            )


            skills = job.get(
                "skills",
                "",
            )


            description = job.get(
                "jobdescription",
                "",
            )


            with st.expander(
                f"{rank}. {title}"
            ):

                st.write(
                    f"**Similarity Score:** "
                    f"{score:.4f}"
                )


                if skills:

                    st.write(
                        f"**Skills:** {skills}"
                    )


                if description:

                    st.write(
                        f"**Description:** {description}"
                    )


    else:

        st.warning(
            "No matching jobs found."
        )


    st.divider()


    # ========================================================
    # CV SUGGESTIONS
    # ========================================================

    st.header("📝 CV Suggestions")


    suggestions = (
        st.session_state.cv_suggestions
    )


    if suggestions:

        # ----------------------------------------------------
        # Missing skills
        # ----------------------------------------------------

        st.subheader(
            "🔧 Missing Skills"
        )


        missing_skills = suggestions.get(
            "missing_skills",
            [],
        )


        if missing_skills:

            for skill in missing_skills:

                st.write(
                    f"• {skill}"
                )

        else:

            st.write(
                "No missing skills identified."
            )


        # ----------------------------------------------------
        # Weak bullet points
        # ----------------------------------------------------

        st.subheader(
            "✏️ Weak Bullet Points"
        )


        weak_bullets = suggestions.get(
            "weak_bullet_points",
            [],
        )


        if weak_bullets:

            for bullet in weak_bullets:

                st.write(
                    f"• {bullet}"
                )

        else:

            st.write(
                "No weak bullet points identified."
            )


        # ----------------------------------------------------
        # Rewritten summary
        # ----------------------------------------------------

        st.subheader(
            "✨ Rewritten Summary"
        )


        rewritten_summary = suggestions.get(
            "rewritten_summary",
            "",
        )


        if rewritten_summary:

            st.write(
                rewritten_summary
            )

        else:

            st.write(
                "No rewritten summary available."
            )


    else:

        st.info(
            "No CV suggestions available."
        )


    st.divider()


    # ========================================================
    # AI CAREER MENTOR
    # ========================================================

    st.header("🤖 AI Career Mentor")


    st.write(
        "Ask questions about careers, jobs, skills, "
        "resumes, interviews, and professional development."
    )


    # ========================================================
    # DISPLAY PREVIOUS MESSAGES
    # ========================================================

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )


    # ========================================================
    # CHAT INPUT
    # ========================================================

    question = st.chat_input(
        "Ask your career question..."
    )


    if question:

        # ====================================================
        # MODULE 5 — GUARDRAIL
        # ====================================================

        allowed, guardrail_message = check_input(
            question
        )


        # ----------------------------------------------------
        # BLOCK INVALID QUESTION
        # ----------------------------------------------------

        if not allowed:

            st.warning(
                f"🛡️ {guardrail_message}"
            )


        else:

            # =================================================
            # RESUME-SPECIFIC QUESTION CHECK
            # =================================================

            if (
                is_resume_specific_question(question)
                and not st.session_state.resume_profile
            ):

                st.warning(
                    "Please upload and analyze your resume "
                    "before asking resume-specific questions."
                )


            else:

                # ------------------------------------------------
                # Save user question
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": question,
                    }
                )


                # ------------------------------------------------
                # Display user question
                # ------------------------------------------------

                with st.chat_message("user"):

                    st.write(
                        question
                    )


                # ------------------------------------------------
                # AI Mentor
                # ------------------------------------------------

                with st.chat_message("assistant"):

                    with st.spinner(
                        "Thinking..."
                    ):

                        try:

                            # ====================================
                            # CALL YOUR ACTUAL RAG CHAIN
                            # ====================================

                            result = ask_mentor(
                                question
                            )


                            # ------------------------------------------------
                            # Get answer
                            # ------------------------------------------------

                            answer = result.get(
                                "answer",
                                "I don't know based on the provided career notes.",
                            )


                            # ------------------------------------------------
                            # Display answer
                            # ------------------------------------------------

                            st.write(
                                answer
                            )


                            # ------------------------------------------------
                            # Display sources
                            # ------------------------------------------------

                            sources = result.get(
                                "sources",
                                []
                            )


                            if sources:

                                st.subheader(
                                    "📚 Sources"
                                )


                                for source in sources:

                                    if isinstance(
                                        source,
                                        dict,
                                    ):

                                        source_name = source.get(
                                            "source",
                                            "Career notes",
                                        )


                                        score = source.get(
                                            "score",
                                            None,
                                        )


                                        if score is not None:

                                            st.write(
                                                f"• {source_name} "
                                                f"(score: {score:.4f})"
                                            )

                                        else:

                                            st.write(
                                                f"• {source_name}"
                                            )

                                    else:

                                        st.write(
                                            f"• {source}"
                                        )


                            # ------------------------------------------------
                            # Save assistant response
                            # ------------------------------------------------

                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": answer,
                                }
                            )


                        except Exception as e:

                            error_message = (
                                f"Sorry, something went wrong: {e}"
                            )


                            st.error(
                                error_message
                            )


                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": error_message,
                                }
                            )