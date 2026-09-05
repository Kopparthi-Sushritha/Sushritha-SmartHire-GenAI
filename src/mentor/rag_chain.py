"""Module 4 — AI Career Mentor (RAG).

A LangChain pipeline that answers career questions grounded in the
career-notes FAISS index.
"""

import sys
import os
from pathlib import Path

import numpy as np
import faiss

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.safety.guardrails import validate_question


# ============================================================
# 1. PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. PROJECT IMPORTS
# ============================================================

from src import config
from src.generate.prompts import MENTOR_SYSTEM_PROMPT


# ============================================================
# 3. LOAD API KEY
# ============================================================

load_dotenv(
    PROJECT_ROOT / ".env.example"
)

api_key = os.getenv(
    config.API_KEY_ENV
)

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY not found in .env.example"
    )


# ============================================================
# 4. MODELS
# ============================================================

CHAT_MODEL = config.CHAT_MODEL

# IMPORTANT:
# The existing career-note FAISS index was created using
# all-MiniLM-L6-v2, so we MUST use the same model for queries.

RAG_EMBED_MODEL = "all-MiniLM-L6-v2"

RAG_EMBED_DIM = 384


# ============================================================
# 5. LOAD EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    RAG_EMBED_MODEL
)


# ============================================================
# 6. LOAD SAVED CAREER-NOTE FAISS INDEX
# ============================================================

index_file = (
    config.NOTES_INDEX_DIR / "index.faiss"
)

if not index_file.exists():
    raise FileNotFoundError(
        "Career-note FAISS index not found:\n"
        + str(index_file)
        + "\n\n"
        "Run Notebook 03 first to create the "
        "career-note FAISS index."
    )

notes_index = faiss.read_index(
    str(index_file)
)


# ============================================================
# 7. LOAD CAREER-NOTE METADATA
# ============================================================

metadata_file = (
    config.NOTES_INDEX_DIR
    / "notes_metadata.npy"
)

if not metadata_file.exists():
    raise FileNotFoundError(
        "Career-note metadata not found:\n"
        + str(metadata_file)
        + "\n\n"
        "Run Notebook 03 first to create "
        "notes_metadata.npy."
    )

metadata = np.load(
    metadata_file,
    allow_pickle=True
)


# ============================================================
# 8. CHECK FAISS + METADATA
# ============================================================

if notes_index.ntotal != len(metadata):
    raise ValueError(
        "FAISS index and metadata size do not match.\n"
        f"FAISS vectors: {notes_index.ntotal}\n"
        f"Metadata records: {len(metadata)}"
    )


if notes_index.d != RAG_EMBED_DIM:
    raise ValueError(
        "Embedding dimension does not match FAISS index.\n"
        f"FAISS dimension: {notes_index.d}\n"
        f"Expected dimension: {RAG_EMBED_DIM}"
    )


# ============================================================
# 9. CREATE GEMINI CHAT MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    google_api_key=api_key,
    temperature=0.2
)


# ============================================================
# 10. CREATE MENTOR PROMPT
# ============================================================

mentor_prompt = ChatPromptTemplate.from_template(
    MENTOR_SYSTEM_PROMPT
)


# ============================================================
# 11. RETRIEVE CAREER NOTES
# ============================================================

def retrieve_notes(
    question: str,
    top_k: int = config.TOP_K_NOTES
):
    """
    Retrieve the top-K career-note chunks from FAISS.
    """

    if not isinstance(question, str):
        raise TypeError(
            "Question must be a string."
        )

    question = question.strip()

    if not question:
        return []


    # --------------------------------------------------------
    # Create question embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Check embedding dimension
    # --------------------------------------------------------

    if query_embedding.shape[1] != notes_index.d:
        raise ValueError(
            "Embedding dimension does not match FAISS index.\n"
            f"Query dimension: {query_embedding.shape[1]}\n"
            f"FAISS dimension: {notes_index.d}\n"
            "Use the same embedding model that was used "
            "to build the FAISS index."
        )


    # --------------------------------------------------------
    # Search FAISS
    # --------------------------------------------------------

    scores, indices = notes_index.search(
        query_embedding,
        top_k
    )


    # --------------------------------------------------------
    # Prepare retrieved notes
    # --------------------------------------------------------

    retrieved = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        if idx >= len(metadata):
            continue

        item = metadata[idx]


        # ====================================================
        # IMPORTANT FIX
        # Handle tuple, list, NumPy array and dictionary
        # ====================================================

        if isinstance(
            item,
            (list, tuple, np.ndarray)
        ) and len(item) >= 2:

            chunk = item[0]
            source = item[1]


        elif isinstance(item, dict):

            chunk = (
                item.get("text")
                or item.get("content")
                or item.get("chunk")
                or ""
            )

            source = (
                item.get("source")
                or item.get("filename")
                or item.get("file")
                or "Unknown source"
            )


        else:
            continue


        # ----------------------------------------------------
        # Convert to strings
        # ----------------------------------------------------

        if chunk is None:
            chunk = ""

        if source is None:
            source = "Unknown source"

        chunk = str(chunk).strip()
        source = str(source).strip()


        # ----------------------------------------------------
        # Skip empty chunks
        # ----------------------------------------------------

        if not chunk:
            continue


        # ----------------------------------------------------
        # Add result
        # ----------------------------------------------------

        retrieved.append(
            {
                "score": float(score),
                "text": chunk,
                "source": source
            }
        )


    return retrieved


# ============================================================
# 12. CAREER MENTOR RAG FUNCTION
# ============================================================

def career_mentor(question: str):
    """
    Answer a career question using retrieved career notes.

    Returns:
        {
            "answer": str,
            "sources": list
        }
    """

    # --------------------------------------------------------
    # Guardrail check
    # --------------------------------------------------------

    allowed, reason = validate_question(
        question
    )

    if not allowed:
        return {
            "answer": "Request rejected: " + reason,
            "sources": []
        }


    # --------------------------------------------------------
    # Retrieve career notes
    # --------------------------------------------------------

    retrieved_notes = retrieve_notes(
        question,
        top_k=config.TOP_K_NOTES
    )


    # --------------------------------------------------------
    # No retrieved notes
    # --------------------------------------------------------

    if not retrieved_notes:

        return {
            "answer": (
                "I don't know based on the "
                "provided career notes."
            ),
            "sources": []
        }


    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for i, item in enumerate(
        retrieved_notes,
        start=1
    ):

        context_parts.append(
            "SOURCE "
            + str(i)
            + ": "
            + item["source"]
            + "\n\n"
            + item["text"]
        )


    context = "\n\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # Create prompt
    # --------------------------------------------------------

    messages = mentor_prompt.format_messages(
        context=context,
        question=question
    )


    # --------------------------------------------------------
    # Generate answer with Gemini
    # --------------------------------------------------------

    response = llm.invoke(
        messages
    )


    # --------------------------------------------------------
    # Extract answer
    # --------------------------------------------------------

    answer = response.content


    if isinstance(answer, list):

        text_parts = []

        for part in answer:

            if isinstance(part, dict):

                text = part.get("text")

                if text:
                    text_parts.append(
                        str(text)
                    )

            elif isinstance(part, str):

                text_parts.append(
                    part
                )


        answer = "\n".join(
            text_parts
        )


    answer = str(answer).strip()


    if not answer:

        answer = (
            "I don't know based on the "
            "provided career notes."
        )


    # --------------------------------------------------------
    # Prepare sources
    # --------------------------------------------------------

    sources = []

    for item in retrieved_notes:

        sources.append(
            {
                "source": item["source"],
                "score": item["score"]
            }
        )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "answer": answer,
        "sources": sources
    }


# ============================================================
# 13. PUBLIC FUNCTION FOR STREAMLIT
# ============================================================

def ask_mentor(question: str):
    """
    Public interface for the Career Mentor.
    """

    return career_mentor(
        question
    )


# ============================================================
# 14. TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SMART HIRE - AI CAREER MENTOR")
    print("=" * 60)

    questions = [
        "What skills should I learn for a career in data science?",
        "How can I improve my career based on the available career notes?",
        "What is the salary of a software engineer at Google?"
    ]

    for question in questions:

        print()
        print("-" * 60)
        print("QUESTION:")
        print(question)
        print("-" * 60)

        try:

            result = career_mentor(
                question
            )

            print()
            print("ANSWER:")
            print(result["answer"])

            print()
            print("SOURCES:")

            if result["sources"]:

                for source in result["sources"]:

                    print(
                        f"- {source['source']} "
                        f"(score={source['score']:.4f})"
                    )

            else:

                print("None")

        except Exception as e:

            print()
            print("ERROR:")
            print(
                type(e).__name__,
                "-",
                e
            )

    print()
    print("=" * 60)
    print("CAREER MENTOR TEST COMPLETED")
    print("=" * 60)