"""
Step 10 — BioLORD Embedding & Synonym Finder
=============================================
For every term in *entity.json* (optionally excluding a specific split):
  1. Computes a BioLORD-2023-M embedding (cached to *embeddings_cache.pkl*).
  2. Finds all pairs of terms whose cosine similarity exceeds SIMILARITY_THRESHOLD.
  3. Writes the synonym mapping to *embedding_synonyms.json*.

Usage:
    python 10_embedding.py [--exclude_split test]
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# ================================
# CONFIGURATION
# ================================

INPUT_FILE          = "entity.json"
CACHE_FILE          = "embeddings_cache.pkl"
SYNONYMS_FILE       = "embedding_synonyms.json"
MODEL_NAME          = "FremyCompany/BioLORD-2023-M"
SIMILARITY_THRESHOLD = 0.9
BATCH_SIZE          = 1000   # Rows processed per similarity-matrix chunk


# ================================
# CACHE HELPERS
# ================================

def _term_key(term: str) -> str:
    """Return a stable SHA-256 hex digest that uniquely identifies (model, term)."""
    raw = f"{MODEL_NAME}::{term}"
    return hashlib.sha256(raw.encode()).hexdigest()


def load_cache(path: str) -> dict:
    """Load the embedding cache from a pickle file, or return an empty dict."""
    p = Path(path)
    if p.exists():
        with open(p, "rb") as f:
            cache = pickle.load(f)
        print(f"[cache] Loaded {len(cache)} cached embeddings from {path}")
        return cache
    return {}


def save_cache(cache: dict, path: str) -> None:
    """Persist the embedding cache to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    print(f"[cache] Saved {len(cache)} embeddings to {path}")


# ================================
# EMBEDDING HELPERS
# ================================

def get_embedding(term: str, cache: dict) -> np.ndarray:
    """
    Retrieve the cached embedding for *term*.

    Raises:
        KeyError: If the term has not been embedded yet.
    """
    key = _term_key(term)
    if key not in cache:
        raise KeyError(f"Embedding not found in cache for term: '{term}'")
    return cache[key]


def similarity(term_a: str, term_b: str, cache: dict) -> float:
    """
    Return the cosine similarity between *term_a* and *term_b*.

    Embeddings are assumed to be L2-normalised, so cosine similarity
    reduces to a plain dot product.
    """
    emb_a = get_embedding(term_a, cache)
    emb_b = get_embedding(term_b, cache)
    return float(emb_a @ emb_b)


def embed_missing_terms(terms: list[str], cache: dict) -> dict:
    """
    Compute and cache embeddings for any *terms* not already in *cache*.

    Returns the updated cache dict.
    """
    missing = [t for t in terms if _term_key(t) not in cache]

    if not missing:
        print("[embed] All embeddings already cached — nothing to compute.")
        return cache

    print(f"[model] Loading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"[embed] Computing embeddings for {len(missing)} new terms...")

    embs = model.encode(missing, show_progress_bar=True, normalize_embeddings=True)
    for term, emb in zip(missing, embs):
        cache[_term_key(term)] = emb

    save_cache(cache, CACHE_FILE)
    return cache


# ================================
# SYNONYM COMPUTATION
# ================================

def compute_synonyms(
    terms: list[str],
    cache: dict,
    threshold: float = SIMILARITY_THRESHOLD,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """
    Find synonym pairs among *terms* using batched cosine-similarity matrix
    multiplication to keep RAM usage bounded.

    Args:
        terms:      List of term strings (embeddings must be in *cache*).
        cache:      Embedding cache dict (term_key → np.ndarray).
        threshold:  Minimum cosine similarity to consider two terms synonyms.
        batch_size: Number of rows processed per matrix chunk.

    Returns:
        A dict mapping each term to a sorted list of
        ``{"value": str, "sim": float}`` dicts for all synonym candidates.
    """
    emb_matrix = np.stack([cache[_term_key(t)] for t in terms])  # (N, D)
    N          = len(terms)
    synonyms: dict = {}

    for i in range(0, N, batch_size):
        end_i      = min(i + batch_size, N)
        batch_embs = emb_matrix[i:end_i]          # (batch, D)
        sim_batch  = batch_embs @ emb_matrix.T    # (batch, N)

        for batch_idx, global_idx in enumerate(range(i, end_i)):
            term         = terms[global_idx]
            sims         = sim_batch[batch_idx]
            valid_indices = np.where(sims >= threshold)[0]

            candidates = [
                {"value": terms[j], "sim": round(float(sims[j]), 4)}
                for j in valid_indices
                if j != global_idx
            ]

            if candidates:
                candidates.sort(key=lambda x: x["sim"], reverse=True)
                synonyms[term] = candidates

    return synonyms


# ================================
# I/O HELPERS
# ================================

def save_synonyms(synonyms: dict, path: str) -> None:
    """Serialise *synonyms* to a JSON file at *path*."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(synonyms, f, ensure_ascii=False, indent=2)
    print(f"[synonyms] Saved {len(synonyms)} entries to {path}")


def load_terms(exclude_split: str | None) -> list[str]:
    """
    Load term names from INPUT_FILE, optionally excluding a specific split.

    Args:
        exclude_split: If provided, terms whose ``split_type`` equals this
                       value are excluded (e.g. ``"test"``).

    Returns:
        A list of term strings.
    """
    with open(INPUT_FILE, encoding="utf-8") as f:
        entities: dict = json.load(f)

    if exclude_split:
        terms = [
            term for term, meta in entities.items()
            if meta.get("split_type") != exclude_split
        ]
    else:
        terms = list(entities.keys())

    print(f"[input] {len(terms)} terms loaded")
    return terms


# ================================
# MAIN
# ================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BioLORD embeddings and synonyms.")
    parser.add_argument(
        "--exclude_split",
        type=str,
        default=None,
        help="Exclude entities with this split_type (e.g. 'test')",
    )
    args = parser.parse_args()

    terms = load_terms(exclude_split=args.exclude_split)

    cache = load_cache(CACHE_FILE)
    cache = embed_missing_terms(terms, cache)

    print(f"[synonyms] Computing synonyms with threshold={SIMILARITY_THRESHOLD} ...")
    synonyms = compute_synonyms(terms, cache, threshold=SIMILARITY_THRESHOLD)
    save_synonyms(synonyms, SYNONYMS_FILE)
    print(f"[synonyms] {len(synonyms)} terms have at least one synonym.")


if __name__ == "__main__":
    main()
