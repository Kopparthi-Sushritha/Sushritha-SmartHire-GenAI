"""Create embeddings for text (jobs, profiles, career notes).

Wrap the Gemini embedding model (config.EMBED_MODEL) so the rest of the project
calls one function instead of the API directly. Build this in notebook 01 while you
test similarity, then move it here.
"""
"""Create embeddings for text (jobs, profiles, career notes).

Wrap the local embedding model so the rest of the project
calls one function instead of the embedding API directly.

The same model must be used when:
- building the FAISS index
- creating query embeddings

Current model:
all-MiniLM-L6-v2

Embedding dimension:
384
"""

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. EMBEDDING MODEL
# ============================================================

EMBED_MODEL = "all-MiniLM-L6-v2"

EMBED_DIM = 384


# ============================================================
# 2. LOAD MODEL
# ============================================================

_embedding_model = SentenceTransformer(EMBED_MODEL)


# ============================================================
# 3. CREATE EMBEDDING
# ============================================================

def embed_text(text: str) -> np.ndarray:
    """Create a normalized embedding for the given text."""

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("Text cannot be empty.")

    embedding = _embedding_model.encode(
        [text],
        normalize_embeddings=True
    )

    return np.asarray(
        embedding[0],
        dtype=np.float32
    )


# ============================================================
# 4. CREATE EMBEDDINGS FOR MULTIPLE TEXTS
# ============================================================

def embed_texts(texts: list[str]) -> np.ndarray:
    """Create normalized embeddings for multiple texts."""

    if not texts:
        return np.empty(
            (0, EMBED_DIM),
            dtype=np.float32
        )

    embeddings = _embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return np.asarray(
        embeddings,
        dtype=np.float32
    )


# ============================================================
# 5. COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray
) -> float:
    """Calculate cosine similarity between two embeddings."""

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# ============================================================
# 6. TEST
# ============================================================

if __name__ == "__main__":

    text1 = "Machine learning and Python"
    text2 = "Python data science and machine learning"

    embedding1 = embed_text(text1)
    embedding2 = embed_text(text2)

    print("Embedding model:", EMBED_MODEL)
    print("Embedding dimension:", len(embedding1))
    print("Embedding shape:", embedding1.shape)

    similarity = cosine_similarity(
        embedding1,
        embedding2
    )

    print("Cosine similarity:", similarity)
