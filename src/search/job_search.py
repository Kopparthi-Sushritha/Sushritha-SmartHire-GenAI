"""Module 2 — Semantic job search.

Embed every job description, store the vectors in a FAISS index (saved under
config.JOBS_INDEX_DIR), embed the candidate profile, and run a top-N similarity
search. Build the index in notebook 02, then move the load/query code here.
"""

"""Module 2 — Semantic job search.

Load the job FAISS index created in notebook 02, embed a candidate
profile using the same local embedding model, and return the top-N
most similar jobs.

The FAISS index is stored under config.JOBS_INDEX_DIR.
"""

import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. FIND PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. IMPORT PROJECT CONFIG
# ============================================================

from src import config


# ============================================================
# 3. EMBEDDING MODEL
# ============================================================

# IMPORTANT:
# This MUST be the same model used to build the FAISS index
# in Notebook 02.

EMBED_MODEL = "all-MiniLM-L6-v2"

EMBED_DIM = 384


# ============================================================
# 4. FAISS INDEX FILE
# ============================================================

INDEX_FILE = (
    config.JOBS_INDEX_DIR /
    "index.faiss"
)


# ============================================================
# 5. JOB METADATA FILE
# ============================================================

JOBS_METADATA_FILE = (
    config.JOBS_INDEX_DIR /
    "jobs.csv"
)


# ============================================================
# 6. LOAD EMBEDDING MODEL
# ============================================================

_embedding_model = SentenceTransformer(
    EMBED_MODEL
)


# ============================================================
# 7. LOAD FAISS INDEX
# ============================================================

if not INDEX_FILE.exists():
    raise FileNotFoundError(
        "Job FAISS index not found:\n"
        f"{INDEX_FILE}\n\n"
        "Run Notebook 02 first to create the job FAISS index."
    )

index = faiss.read_index(
    str(INDEX_FILE)
)


# ============================================================
# 8. LOAD JOB METADATA
# ============================================================

if not JOBS_METADATA_FILE.exists():
    raise FileNotFoundError(
        "Job metadata file not found:\n"
        f"{JOBS_METADATA_FILE}\n\n"
        "Run Notebook 02 first to create jobs.csv."
    )

jobs_df = pd.read_csv(
    JOBS_METADATA_FILE
)


# ============================================================
# 9. VERIFY INDEX AND JOB DATA
# ============================================================

if index.ntotal != len(jobs_df):
    raise ValueError(
        "FAISS index and job metadata do not match.\n"
        f"FAISS vectors: {index.ntotal}\n"
        f"Jobs in CSV: {len(jobs_df)}"
    )


if index.d != EMBED_DIM:
    raise ValueError(
        "FAISS embedding dimension does not match the "
        "current embedding model.\n"
        f"FAISS dimension: {index.d}\n"
        f"Expected dimension: {EMBED_DIM}"
    )


# ============================================================
# 10. CREATE CANDIDATE EMBEDDING
# ============================================================

def embed_candidate(profile: str) -> np.ndarray:
    """Create an embedding for a candidate profile."""

    if not isinstance(profile, str):
        raise TypeError(
            "Candidate profile must be a string."
        )

    profile = profile.strip()

    if not profile:
        raise ValueError(
            "Candidate profile cannot be empty."
        )

    embedding = _embedding_model.encode(
        [profile],
        normalize_embeddings=True
    )

    return np.asarray(
        embedding,
        dtype=np.float32
    )


# ============================================================
# 11. SEARCH FOR SIMILAR JOBS
# ============================================================

def search_jobs(
    candidate_profile: str,
    top_n: int = config.TOP_N_JOBS
):
    """Return the top-N jobs most similar to a candidate profile."""

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than 0."
        )

    # Do not request more jobs than exist
    top_n = min(
        top_n,
        index.ntotal
    )

    # Create candidate embedding
    candidate_embedding = embed_candidate(
        candidate_profile
    )

    # Search FAISS
    scores, indices = index.search(
        candidate_embedding,
        top_n
    )

    results = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx < 0:
            continue

        job = jobs_df.iloc[idx]

        result = {
            "rank": rank,
            "similarity_score": float(score),
            "jobtitle": (
                str(job["jobtitle"])
                if pd.notna(job["jobtitle"])
                else ""
            ),
            "skills": (
                str(job["skills"])
                if pd.notna(job["skills"])
                else ""
            ),
            "jobdescription": (
                str(job["jobdescription"])
                if pd.notna(job["jobdescription"])
                else ""
            )
        }

        results.append(result)

    return results


# ============================================================
# 12. SEARCH AND RETURN DATAFRAME
# ============================================================

def search_jobs_df(
    candidate_profile: str,
    top_n: int = config.TOP_N_JOBS
):
    """Return top matching jobs as a pandas DataFrame."""

    results = search_jobs(
        candidate_profile,
        top_n
    )

    return pd.DataFrame(results)


# ============================================================
# 13. TEST THE MODULE
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SMART HIRE - SEMANTIC JOB SEARCH")
    print("=" * 70)

    print("\nFAISS index:")
    print(INDEX_FILE)

    print("\nTotal jobs in index:")
    print(index.ntotal)

    print("\nEmbedding model:")
    print(EMBED_MODEL)

    print("\nEmbedding dimension:")
    print(EMBED_DIM)

    # --------------------------------------------------------
    # Candidate profile for testing
    # --------------------------------------------------------

    candidate_profile = """
    Skills: Python, Machine Learning, Data Science, SQL,
    Pandas, NumPy, Scikit-learn, TensorFlow.

    Experience: Experience in data analysis, machine learning,
    and building predictive models.

    Education: Computer Science / Information Technology.

    Target Role: Machine Learning Engineer / AI Engineer.
    """

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    print(
        f"\nSearching for top {config.TOP_N_JOBS} matching jobs..."
    )

    results = search_jobs(
        candidate_profile,
        config.TOP_N_JOBS
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP MATCHING JOBS")
    print("=" * 70)

    for result in results:

        print("\n" + "-" * 70)

        print(
            "Rank:",
            result["rank"]
        )

        print(
            "Similarity Score:",
            f"{result['similarity_score']:.4f}"
        )

        print(
            "Job Title:",
            result["jobtitle"]
        )

        print(
            "Skills:",
            result["skills"]
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SEMANTIC JOB SEARCH COMPLETED")
    print("=" * 70)

    print(
        "Total jobs in FAISS:",
        index.ntotal
    )

    print(
        "Results returned:",
        len(results)
    )