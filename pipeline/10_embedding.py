import argparse
import json
import hashlib
import pickle
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT_FILE = "entity.json"
CACHE_FILE = "embeddings_cache.pkl"
SYNONYMS_FILE = "embedding_synonyms.json"
MODEL_NAME = "FremyCompany/BioLORD-2023-M"
SIMILARITY_THRESHOLD = 0.9

def _term_key(term: str) -> str:
    raw = f"{MODEL_NAME}::{term}"
    return hashlib.sha256(raw.encode()).hexdigest()


def load_entities(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

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

def compute_synonyms(
    terms: list[str],
    cache: dict,
    threshold: float = SIMILARITY_THRESHOLD,
    batch_size: int = 1000,
) -> dict:
    emb_matrix = np.stack([cache[_term_key(t)] for t in terms])  # (N, D)
    N = len(terms)
    synonyms: dict = {}

    # process the matrix not all at once, but in batches of batch_size (e.g., 1000) rows
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        batch_embs = emb_matrix[i:end_i]  # (batch_size, D)

        # calculate similarity for this batch (RAM usage fix)
        # sim_batch dimesion: (batch_size, N)
        sim_batch = batch_embs @ emb_matrix.T

        for batch_idx, global_idx in enumerate(range(i, end_i)):
            term = terms[global_idx]
            sims = sim_batch[batch_idx]
            
            # filter the indices that exceed the threshold using NumPy's fast 'where' function
            valid_indices = np.where(sims >= threshold)[0]
            
            candidates = []
            for j in valid_indices:
                if global_idx == j:
                    continue
                candidates.append({
                    "value": terms[j], 
                    "sim": round(float(sims[j]), 4)
                })

            if candidates:
                candidates.sort(key=lambda x: x["sim"], reverse=True)
                synonyms[term] = candidates

    return synonyms

def save_synonyms(synonyms: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(synonyms, f, ensure_ascii=False, indent=2)
    print(f"[synonyms] Saved {len(synonyms)} entries to {path}")
    
def get_embedding(term: str, cache: dict) -> np.ndarray:
    key = _term_key(term)
    if key not in cache:
        raise KeyError(f"'{term}' not found in cache")
    return cache[key]

def similarity(term_a: str, term_b: str, cache: dict) -> float:
    emb_a = get_embedding(term_a, cache)
    emb_b = get_embedding(term_b, cache)
    return float(emb_a @ emb_b)  # L2-normalised → cosine similarity

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exclude_split",
        type=str,
        default=None,
        help="Exclude entities with this split_type (e.g. test)"
    )
    args = parser.parse_args()
    
    entities = load_entities(INPUT_FILE)
    
    terms = []
    if args.exclude_split:
        terms = [
            term for term, meta in entities.items()
            if meta.get("split_type") != args.exclude_split
        ]
    else:
        terms = list(entities.keys())
    print(f"[input] {len(terms)} terms loaded")

    cache = load_cache(CACHE_FILE)
    missing = [t for t in terms if _term_key(t) not in cache]

    if missing:
        print(f"[model] Loading {MODEL_NAME} ...")
        model = SentenceTransformer(MODEL_NAME)
        print(f"[embed] Computing embeddings for {len(missing)} new terms...")
        embs = model.encode(missing, show_progress_bar=True, normalize_embeddings=True)
        for term, emb in zip(missing, embs):
            cache[_term_key(term)] = emb
        save_cache(cache, CACHE_FILE)
    else:
        print("[embed] All embeddings already cached, nothing to do.")
    
    print(f"[synonyms] Computing synonyms with threshold={SIMILARITY_THRESHOLD} ...")
    synonyms = compute_synonyms(terms, cache, threshold=SIMILARITY_THRESHOLD)
    save_synonyms(synonyms, SYNONYMS_FILE)
    print(f"[synonyms] {len(synonyms)} terms have at least one synonym.")
    

if __name__ == "__main__":
    main()