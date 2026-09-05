"""Answer-quality and retrieval evaluation.

Evaluate the SmartHire system on a small test set:

1. Retrieval relevance:
   Check whether relevant jobs appear in the top-N results.

2. Answer quality:
   Check mentor answers for correctness, grounding, and helpfulness.

3. Hallucination check:
   Check whether the mentor refuses when the answer is not
   supported by the career notes.

Results are written to:
reports/answer_quality.md
"""

import sys
from pathlib import Path


# ============================================================
# 1. FIND PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. IMPORT PROJECT MODULES
# ============================================================

from src import config
from src.search.job_search import search_jobs
from src.mentor.rag_chain import career_mentor


# ============================================================
# 3. REPORT LOCATION
# ============================================================

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_FILE = REPORT_DIR / "answer_quality.md"


# ============================================================
# 4. RETRIEVAL TEST SET
# ============================================================

RETRIEVAL_TESTS = [
    {
        "name": "Machine Learning Candidate",
        "profile": """
        Skills: Python, Machine Learning, Data Science, SQL,
        Pandas, NumPy, Scikit-learn, TensorFlow.

        Experience: Experience in data analysis, machine learning,
        predictive modelling and building machine learning models.

        Education: Computer Science / Information Technology.

        Target Role: Machine Learning Engineer / AI Engineer.
        """,
        "expected_keywords": [
            "machine learning",
            "data scientist",
            "machine learning engineer",
            "artificial intelligence"
        ]
    },

    {
        "name": "Data Science Candidate",
        "profile": """
        Skills: Python, SQL, Statistics, Pandas, NumPy,
        Machine Learning, Data Analysis, Data Visualization.

        Experience: Experience working with datasets,
        statistical analysis, data cleaning and predictive models.

        Education: Computer Science / Data Science.

        Target Role: Data Scientist / Data Analyst.
        """,
        "expected_keywords": [
            "data scientist",
            "data analyst",
            "data science",
            "machine learning"
        ]
    },

    {
        "name": "Software Engineer Candidate",
        "profile": """
        Skills: Python, Java, Data Structures, Algorithms,
        Object Oriented Programming, SQL.

        Experience: Experience developing software applications,
        solving programming problems and working with software systems.

        Education: Computer Science.

        Target Role: Software Engineer / Backend Developer.
        """,
        "expected_keywords": [
            "software engineer",
            "software developer",
            "backend",
            "developer"
        ]
    }
]


# ============================================================
# 5. MENTOR ANSWER TEST SET
# ============================================================

MENTOR_TESTS = [
    {
        "question": "What skills should I learn for a career in data science?",
        "type": "answer_quality"
    },

    {
        "question": "How can I improve my career based on the available career notes?",
        "type": "answer_quality"
    }
]


# ============================================================
# 6. HALLUCINATION TEST
# ============================================================

HALLUCINATION_TEST = {
    "question": "What is the salary of a software engineer at Google?",
    "expected_refusal": (
        "I don't know based on the provided career notes."
    )
}


# ============================================================
# 7. CHECK RETRIEVAL RELEVANCE
# ============================================================

def check_retrieval_relevance(
    results,
    expected_keywords
):
    """Check whether at least one retrieved job is relevant."""

    for result in results:

        job_title = str(
            result.get("jobtitle", "")
        ).lower()

        skills = str(
            result.get("skills", "")
        ).lower()

        description = str(
            result.get("jobdescription", "")
        ).lower()

        combined_text = (
            job_title + " " +
            skills + " " +
            description
        )

        for keyword in expected_keywords:

            if keyword.lower() in combined_text:
                return True

    return False


# ============================================================
# 8. CALCULATE RETRIEVAL HIT RATE
# ============================================================

def evaluate_retrieval():
    """Evaluate job retrieval using the sample profiles."""

    results = []

    hits = 0

    for test in RETRIEVAL_TESTS:

        retrieved_jobs = search_jobs(
            test["profile"],
            config.TOP_N_JOBS
        )

        hit = check_retrieval_relevance(
            retrieved_jobs,
            test["expected_keywords"]
        )

        if hit:
            hits += 1

        results.append({
            "name": test["name"],
            "hit": hit,
            "jobs": retrieved_jobs
        })

    hit_rate = (
        hits / len(RETRIEVAL_TESTS)
        if RETRIEVAL_TESTS
        else 0
    )

    return results, hit_rate


# ============================================================
# 9. CHECK MENTOR ANSWER
# ============================================================

def check_answer_quality(
    answer,
    sources=None,
    question=None
):
    """
    Check mentor answer quality.

    The checks are intentionally simple:
    - correctness: meaningful answer or correct refusal
    - grounding: answer refers to notes/context OR has retrieved sources
    - helpfulness: meaningful non-error answer
    """

    if not answer:

        return {
            "correctness": False,
            "grounding": False,
            "helpfulness": False
        }

    answer_text = str(answer).strip()
    answer_lower = answer_text.lower()

    if not answer_text:

        return {
            "correctness": False,
            "grounding": False,
            "helpfulness": False
        }


    # --------------------------------------------------------
    # Error detection
    # --------------------------------------------------------

    has_error = (
        answer_lower.startswith("error:")
        or "traceback" in answer_lower
        or "nonetype" in answer_lower
        or "exception" in answer_lower
    )


    # --------------------------------------------------------
    # Correct refusal is considered correct
    # --------------------------------------------------------

    refusal = (
        "i don't know based on the provided career notes."
        in answer_lower
    )


    # --------------------------------------------------------
    # Correctness
    # --------------------------------------------------------

    if refusal:

        correctness = True

    else:

        correctness = (
            len(answer_text) >= 30
            and not has_error
        )


    # --------------------------------------------------------
    # Grounding
    # --------------------------------------------------------

    grounding_words = [
        "according",
        "career notes",
        "provided notes",
        "notes",
        "source",
        "based on"
    ]

    mentions_grounding = any(
        word in answer_lower
        for word in grounding_words
    )


    has_sources = bool(sources)


    grounding = (
        mentions_grounding
        or has_sources
        or refusal
    )


    # --------------------------------------------------------
    # Helpfulness
    # --------------------------------------------------------

    helpfulness = (
        len(answer_text) >= 30
        and not has_error
    )


    return {
        "correctness": correctness,
        "grounding": grounding,
        "helpfulness": helpfulness
    }


# ============================================================
# 10. EVALUATE MENTOR ANSWERS
# ============================================================

def evaluate_mentor_answers():
    """Evaluate normal mentor questions."""

    results = []

    for test in MENTOR_TESTS:

        try:

            response = career_mentor(
                test["question"]
            )

            if response is None:
                raise ValueError(
                    "career_mentor returned None"
                )


            answer = response.get(
                "answer",
                ""
            )


            sources = response.get(
                "sources",
                []
            )


            quality = check_answer_quality(
                answer,
                sources=sources,
                question=test["question"]
            )


            results.append({
                "question": test["question"],
                "answer": answer,
                "sources": sources,
                "quality": quality
            })


        except Exception as e:

            results.append({
                "question": test["question"],
                "answer": f"ERROR: {e}",
                "sources": [],
                "quality": {
                    "correctness": False,
                    "grounding": False,
                    "helpfulness": False
                }
            })


    return results


# ============================================================
# 11. HALLUCINATION TEST
# ============================================================

def evaluate_hallucination():
    """Check whether the mentor refuses unsupported questions."""

    try:

        response = career_mentor(
            HALLUCINATION_TEST["question"]
        )


        if response is None:

            return {
                "question": HALLUCINATION_TEST["question"],
                "answer": "ERROR: career_mentor returned None",
                "passed": False
            }


        answer = str(
            response.get(
                "answer",
                ""
            )
        ).strip()


        expected = HALLUCINATION_TEST[
            "expected_refusal"
        ]


        refusal_found = (
            expected.lower()
            in answer.lower()
        )


        return {
            "question": HALLUCINATION_TEST["question"],
            "answer": answer,
            "passed": refusal_found
        }


    except Exception as e:

        return {
            "question": HALLUCINATION_TEST["question"],
            "answer": f"ERROR: {e}",
            "passed": False
        }


# ============================================================
# 12. CREATE MARKDOWN REPORT
# ============================================================

def create_report(
    retrieval_results,
    retrieval_hit_rate,
    mentor_results,
    hallucination_result
):
    """Create the answer_quality.md report."""

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    report = []


    # ========================================================
    # Retrieval Relevance
    # ========================================================

    report.append(
        "# SmartHire Answer Quality Evaluation"
    )

    report.append("")

    report.append(
        "## 1. Retrieval Relevance"
    )

    report.append("")

    report.append(
        f"**Hit rate:** "
        f"{retrieval_hit_rate * 100:.2f}%"
    )

    report.append("")

    report.append(
        "A retrieval test is counted as a hit when at least "
        "one of the top-N retrieved jobs contains a relevant "
        "job title, skill, or description keyword."
    )

    report.append("")

    report.append(
        "| Test Profile | Hit | Top Job | Score |"
    )

    report.append(
        "|---|---|---|---:|"
    )


    for result in retrieval_results:

        jobs = result["jobs"]


        if jobs:

            top_job = jobs[0]

            top_title = str(
                top_job["jobtitle"]
            ).replace("|", "/")

            top_score = float(
                top_job["similarity_score"]
            )

        else:

            top_title = "No results"
            top_score = 0.0


        hit_text = (
            "PASS"
            if result["hit"]
            else "FAIL"
        )


        report.append(
            f"| {result['name']} | "
            f"{hit_text} | "
            f"{top_title} | "
            f"{top_score:.4f} |"
        )


    report.append("")
    report.append("---")
    report.append("")


    # ========================================================
    # Mentor Answer Quality
    # ========================================================

    report.append(
        "## 2. Mentor Answer Quality"
    )

    report.append("")

    report.append(
        "The mentor answers are checked for basic "
        "correctness, grounding, and helpfulness indicators."
    )

    report.append("")

    report.append(
        "| Question | Correctness | Grounding | Helpfulness |"
    )

    report.append(
        "|---|---|---|---|"
    )


    for result in mentor_results:

        quality = result["quality"]


        correctness = (
            "PASS"
            if quality["correctness"]
            else "FAIL"
        )


        grounding = (
            "PASS"
            if quality["grounding"]
            else "FAIL"
        )


        helpfulness = (
            "PASS"
            if quality["helpfulness"]
            else "FAIL"
        )


        question = result["question"].replace(
            "|",
            "/"
        )


        report.append(
            f"| {question} | "
            f"{correctness} | "
            f"{grounding} | "
            f"{helpfulness} |"
        )


    report.append("")

    report.append(
        "### Mentor Answers"
    )

    report.append("")


    for number, result in enumerate(
        mentor_results,
        start=1
    ):

        report.append(
            f"#### Question {number}"
        )

        report.append("")

        report.append(
            f"**Question:** {result['question']}"
        )

        report.append("")

        report.append(
            "**Answer:**"
        )

        report.append("")

        report.append(
            str(result["answer"])
        )

        report.append("")


        if result["sources"]:

            report.append(
                "**Retrieved sources:**"
            )

            for source in result["sources"]:

                if isinstance(source, dict):

                    source_name = source.get(
                        "source",
                        "Unknown"
                    )

                    score = source.get(
                        "score",
                        0
                    )

                    report.append(
                        f"- {source_name} "
                        f"(score: {float(score):.4f})"
                    )

                else:

                    report.append(
                        f"- {source}"
                    )

        else:

            report.append(
                "**Retrieved sources:** None"
            )


        report.append("")


    report.append("---")
    report.append("")


    # ========================================================
    # Hallucination Check
    # ========================================================

    report.append(
        "## 3. Hallucination Check"
    )

    report.append("")

    report.append(
        "The mentor should refuse to answer when the "
        "requested information is not supported by the "
        "career notes."
    )

    report.append("")


    hallucination_status = (
        "PASS"
        if hallucination_result["passed"]
        else "FAIL"
    )


    report.append(
        f"**Hallucination test: {hallucination_status}**"
    )

    report.append("")

    report.append(
        f"**Question:** "
        f"{hallucination_result['question']}"
    )

    report.append("")

    report.append(
        "**Mentor response:**"
    )

    report.append("")

    report.append(
        hallucination_result["answer"]
    )

    report.append("")

    report.append("---")
    report.append("")


    # ========================================================
    # Final Summary
    # ========================================================

    report.append(
        "## 4. Summary"
    )

    report.append("")

    report.append(
        f"- Retrieval hit rate: "
        f"{retrieval_hit_rate * 100:.2f}%"
    )


    mentor_passes = sum(
        1
        for result in mentor_results
        if all(result["quality"].values())
    )


    report.append(
        f"- Mentor tests passing all basic quality checks: "
        f"{mentor_passes}/{len(mentor_results)}"
    )


    report.append(
        f"- Hallucination refusal: "
        f"{'PASS' if hallucination_result['passed'] else 'FAIL'}"
    )

    report.append("")

    report.append(
        "Note: These are small automated evaluation checks. "
        "Correctness and helpfulness should also be reviewed "
        "manually for a reliable assessment."
    )


    REPORT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8"
    )


    return REPORT_FILE


# ============================================================
# 13. MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("SMART HIRE - SYSTEM EVALUATION")
    print("=" * 70)


    # --------------------------------------------------------
    # Retrieval evaluation
    # --------------------------------------------------------

    print(
        "\n1. Evaluating retrieval relevance..."
    )


    retrieval_results, retrieval_hit_rate = (
        evaluate_retrieval()
    )


    print(
        f"Retrieval hit rate: "
        f"{retrieval_hit_rate * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Mentor evaluation
    # --------------------------------------------------------

    print(
        "\n2. Evaluating mentor answers..."
    )


    mentor_results = evaluate_mentor_answers()


    for result in mentor_results:

        print(
            "\nQuestion:",
            result["question"]
        )

        print(
            "Answer:",
            result["answer"]
        )


    # --------------------------------------------------------
    # Hallucination evaluation
    # --------------------------------------------------------

    print(
        "\n3. Checking hallucination refusal..."
    )


    hallucination_result = (
        evaluate_hallucination()
    )


    print(
        "Hallucination test:",
        "PASS"
        if hallucination_result["passed"]
        else "FAIL"
    )


    # --------------------------------------------------------
    # Create report
    # --------------------------------------------------------

    report_file = create_report(
        retrieval_results,
        retrieval_hit_rate,
        mentor_results,
        hallucination_result
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "EVALUATION COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "\nReport saved to:"
    )

    print(
        report_file
    )


# ============================================================
# 14. RUN
# ============================================================

if __name__ == "__main__":
    main()