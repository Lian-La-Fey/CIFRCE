import argparse
import hashlib
import json
import pickle

import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer


# ================================
# CACHE HELPERS
# ================================

def _term_key(term: str, model_name: str) -> str:
    raw = f"{model_name}::{term}"
    return hashlib.sha256(raw.encode()).hexdigest()

def load_cache(path: str) -> dict:
    p = Path(path)
    if p.exists():
        with open(p, "rb") as f:
            cache = pickle.load(f)
        print(f"[cache] Loaded {len(cache)} cached embeddings from {path}")
        return cache
    return {}

def save_cache(cache: dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    print(f"[cache] Saved {len(cache)} embeddings to {path}")


# ================================
# EMBEDDING HELPERS
# ================================

def get_embedding(term: str, cache: dict, model_name: str) -> np.ndarray:
    key = _term_key(term, model_name)
    if key not in cache:
        raise KeyError(f"Embedding not found in cache for term: '{term}'")
    return cache[key]


def similarity(term_a: str, term_b: str, cache: dict, model_name: str) -> float:
    emb_a = get_embedding(term_a, cache, model_name)
    emb_b = get_embedding(term_b, cache, model_name)
    return float(emb_a @ emb_b)


def embed_missing_terms(
    terms: list[str],
    cache: dict,
    model_name: str,
    cache_file: str,
) -> dict:
    missing = [t for t in terms if _term_key(t, model_name) not in cache]

    if not missing:
        print("[embed] All embeddings already cached — nothing to compute.")
        return cache

    print(f"[model] Loading {model_name} ...")
    model = SentenceTransformer(model_name)

    print(f"[embed] Computing embeddings for {len(missing)} new terms...")
    embs = model.encode(
        missing,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    for term, emb in zip(missing, embs):
        cache[_term_key(term, model_name)] = emb

    save_cache(cache, cache_file)
    return cache


# ================================
# SYNONYM COMPUTATION
# ================================

def compute_synonyms(
    terms: list[str],
    cache: dict,
    model_name: str,
    threshold: float,
    batch_size: int,
) -> dict:
    emb_matrix = np.stack(
        [cache[_term_key(t, model_name)] for t in terms]
    )
    N = len(terms)
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(synonyms, f, ensure_ascii=False, indent=2)
    print(f"[synonyms] Saved {len(synonyms)} entries to {path}")


def load_terms(input_file: str, exclude_split: str | None) -> list[str]:
    with open(input_file, encoding="utf-8") as f:
        entities = json.load(f)

    if exclude_split:
        terms = [
            term
            for term, meta in entities.items()
            if meta.get("split_type") != exclude_split
        ]
    else:
        terms = list(entities.keys())

    print(f"[input] {len(terms)} terms loaded")
    return terms


# ================================
# MAIN
# ================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="entity.json")
    parser.add_argument("--cache_file", default="embeddings_cache.pkl")
    parser.add_argument("--synonyms_file", default="embedding_synonyms.json")
    
    parser.add_argument("--model_name", default="FremyCompany/BioLORD-2023-M")
    parser.add_argument("--similarity_threshold", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=1000,)
    parser.add_argument("--exclude_split", type=str, default=None)
    
    args = parser.parse_args()

    terms = load_terms(
        input_file=args.input_file,
        exclude_split=args.exclude_split,
    )

    cache = load_cache(args.cache_file)
    cache = embed_missing_terms(
        terms=terms,
        cache=cache,
        model_name=args.model_name,
        cache_file=args.cache_file,
    )

    print(
        f"[synonyms] Computing synonyms "
        f"with threshold={args.similarity_threshold} ..."
    )

    synonyms = compute_synonyms(
        terms=terms,
        cache=cache,
        model_name=args.model_name,
        threshold=args.similarity_threshold,
        batch_size=args.batch_size,
    )

    save_synonyms(synonyms, args.synonyms_file)
    print(f"[synonyms] {len(synonyms)} terms have at least one synonym.")

if __name__ == "__main__":
    main()
