import json
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import spacy
import torch

from scispacy.linking import EntityLinker
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ================================
# CONFIGURATION
# ================================

INPUT_JSON       = "unique_entity_counts_test.json"
ENTITY_JSON      = "entity.json"
ENTITY_RULE_JSON = "entity_rule.json"

SPACY_MODEL_NAME    = "en_core_sci_md"
EMBEDDING_MODEL_NAME = "FremyCompany/BioLORD-2023-M"
UMLS_THRESHOLD      = 0.95   # SciSpacy linker candidate threshold
UMLS_TOP_K          = 20     # SciSpacy linker top-k candidates
MATCH_THRESHOLD     = 0.9125 # Minimum cosine similarity to accept a UMLS match

MAX_WORKERS   = 8
SAVE_INTERVAL = 1000

STOP_WORDS = {
    "of", "the", "a", "an", "in", "on", "at", "to", "for",
    "with", "by", "from", "and", "or", "is", "are", "was",
    "were", "be", "been", "into", "within", "without", "as",
}


# ================================
# GLOBAL STATE  (models + caches)
# ================================

print("Loading the SciSpacy model. This may take a moment...")
nlp = spacy.load(SPACY_MODEL_NAME)
nlp.add_pipe(
    "scispacy_linker",
    config={
        "resolve_abbreviations": True,
        "linker_name": "umls",
        "threshold": UMLS_THRESHOLD,
        "k": UMLS_TOP_K,
    },
)
LINKER: EntityLinker = nlp.get_pipe("scispacy_linker")
print("SciSpacy model loaded successfully.")

print("Loading the Embedding model...")
MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Embedding model loaded successfully.")

# Runtime caches — populated from disk in load_caches()
ENTITY_CACHE: dict[str, dict] = {}
ENTITY_RULE_CACHE: dict[str, dict] = {}

# Thread-safety locks
ENTITY_CACHE_LOCK      = threading.Lock()
ENTITY_RULE_CACHE_LOCK = threading.Lock()
SAVE_LOCK              = threading.Lock()


# ================================
# CACHE MANAGEMENT
# ================================

def load_caches() -> None:
    global ENTITY_CACHE, ENTITY_RULE_CACHE

    if Path(ENTITY_JSON).exists():
        with open(ENTITY_JSON, "r", encoding="utf-8") as f:
            ENTITY_CACHE = json.load(f)
        print(f"{ENTITY_JSON} loaded ({len(ENTITY_CACHE)} records)")

    if Path(ENTITY_RULE_JSON).exists():
        with open(ENTITY_RULE_JSON, "r", encoding="utf-8") as f:
            ENTITY_RULE_CACHE = json.load(f)
        print(f"{ENTITY_RULE_JSON} loaded ({len(ENTITY_RULE_CACHE)} records)")


def save_caches() -> None:
    with SAVE_LOCK:
        with ENTITY_CACHE_LOCK:
            snapshot_entity = dict(ENTITY_CACHE)
        with ENTITY_RULE_CACHE_LOCK:
            snapshot_rule = dict(ENTITY_RULE_CACHE)

        with open(ENTITY_JSON, "w", encoding="utf-8") as f:
            json.dump(snapshot_entity, f, indent=2, ensure_ascii=False)
        with open(ENTITY_RULE_JSON, "w", encoding="utf-8") as f:
            json.dump(snapshot_rule, f, indent=2, ensure_ascii=False)


# ================================
# TEXT HELPERS
# ================================

def remove_stopwords(text: str) -> str:
    tokens = text.lower().split()
    return " ".join(t for t in tokens if t not in STOP_WORDS)


def make_result_dict(term: str, res: dict | None, split_type: str) -> dict:
    if res:
        return {
            "term":           term,
            "found":          True,
            "source":         res["source"],
            "id":             res["id"],
            "preferred_name": res["name"],
            "vocabulary":     res["vocabulary"],
            "semantic_types": res["semantic_types"],
            "split_type":     split_type,
        }
    return {
        "term":           term,
        "found":          False,
        "source":         None,
        "id":             None,
        "preferred_name": None,
        "vocabulary":     None,
        "semantic_types": [],
        "split_type":     split_type,
    }


# ================================
# SCISPACY RESOLUTION
# ================================

def resolve_scispacy(term: str) -> dict | None:
    doc = nlp(term)
    if not doc.ents:
        return None

    best_match_cui: str | None = None
    best_score = -1.0

    for ent in doc.ents:
        if ent._.kb_ents:
            cui, score = ent._.kb_ents[0]
            if score > best_score:
                best_score = score
                best_match_cui = cui

    if not best_match_cui:
        return None

    concept = LINKER.kb.cui_to_entity[best_match_cui]
    matched_name = concept.canonical_name

    term_emb    = MODEL.encode(term,         convert_to_tensor=True)
    matched_emb = MODEL.encode(matched_name, convert_to_tensor=True)
    similarity  = util.cos_sim(term_emb, matched_emb).cpu().numpy()[0][0]

    if similarity < MATCH_THRESHOLD:
        return None  # Reject low-confidence / off-topic UMLS matches

    return {
        "source": "UMLS - Scispacy",
        "id": concept.concept_id,
        "name": matched_name,
        "vocabulary": "UMLS",
        "semantic_types": concept.types,
    }


# ================================
# THREAD-SAFE TERM RESOLVER
# ================================

def resolve_term_cached(term: str, split_type: str) -> dict:
    key = term.lower().strip()

    if key in ENTITY_CACHE:
        return ENTITY_CACHE[key]

    result = resolve_scispacy(term)
    result = make_result_dict(term, result, split_type)

    with ENTITY_CACHE_LOCK:
        if key not in ENTITY_CACHE:   # double-checked locking
            ENTITY_CACHE[key] = result

    return result


# ================================
# TOKENISATION / PARTITION LOGIC
# ================================

def get_partitions(words: list[str]) -> list[list[str]]:
    if not words:
        return []
    if len(words) > 7:
        return [words]

    partitions = [[words[0]]]
    for word in words[1:]:
        new_partitions = []
        for p in partitions:
            merged  = p.copy(); merged[-1] += " " + word
            split   = p.copy(); split.append(word)
            new_partitions.extend([merged, split])
        partitions = new_partitions

    return partitions


def resolve_phrase_combinations(phrase: str, split_type: str) -> dict:
    key = phrase.lower().strip()

    if key in ENTITY_RULE_CACHE:
        return ENTITY_RULE_CACHE[key]

    tokens         = key.split()
    all_partitions = get_partitions(tokens)
    unique_tokens  = {token for partition in all_partitions for token in partition}

    # Pre-resolve tokens that are not yet in the entity cache
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as token_executor:
        futures = {
            token_executor.submit(resolve_term_cached, token, split_type): token
            for token in unique_tokens
            if token.lower().strip() not in ENTITY_CACHE
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                tqdm.write(f"[Token Error] {futures[future]}: {e}")

    rule_data = {"rules": all_partitions, "split": split_type}

    with ENTITY_RULE_CACHE_LOCK:
        if key not in ENTITY_RULE_CACHE:
            ENTITY_RULE_CACHE[key] = rule_data

    return rule_data


# ================================
# PER-ENTITY WORKER
# ================================

def process_single_entity(entry: dict) -> str | None:
    name       = entry.get("name", "").strip().lower()
    field_name = entry.get("field_name", "unknown")

    if not name or field_name == "measurement":
        return None

    clean_name = remove_stopwords(name)
    if not clean_name:
        return None

    resolve_phrase_combinations(clean_name, split_type="test")
    return clean_name


# ================================
# MAIN PIPELINE
# ================================

def process_unique_entities(input_path: str) -> None:
    load_caches()

    with open(input_path, "r", encoding="utf-8") as f:
        entities = json.load(f)

    processed_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_entity, entry): entry
            for entry in entities
        }

        with tqdm(total=len(futures), desc="Processing entities") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    entry = futures[future]
                    tqdm.write(f"[Entity Error] {entry.get('name', '?')}: {e}")
                finally:
                    processed_count += 1
                    pbar.update(1)

                    if processed_count % SAVE_INTERVAL == 0:
                        save_caches()

    save_caches()
    print(f"\nCompleted. {processed_count} entities processed.")
    print(f"Entity cache: {len(ENTITY_CACHE)} | Rule cache: {len(ENTITY_RULE_CACHE)}")


def main() -> None:
    process_unique_entities(INPUT_JSON)


if __name__ == "__main__":
    main()
