import json
import threading
import torch
import spacy

from scispacy.linking import EntityLinker
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

# ====================================
# Model loaders
# ====================================

def load_scispacy_model() -> tuple[spacy.Language, EntityLinker]:
    print("Loading the SciSpacy model. This may take a moment...")
    nlp = spacy.load("en_core_sci_md")
    nlp.add_pipe(
        "scispacy_linker",
        config={
            "resolve_abbreviations": True,
            "linker_name": "umls",
            "threshold": 0.95,
            "k": 20,
        },
    )
    linker: EntityLinker = nlp.get_pipe("scispacy_linker")
    print("The SciSpacy model has been successfully loaded.")
    return nlp, linker


def load_embedding_model() -> SentenceTransformer:
    print("Loading the Embedding model...")
    model = SentenceTransformer("FremyCompany/BioLORD-2023-M")
    print("The Embedding model has been successfully loaded.")
    return model


# ====================================
# SCISPACY MODEL & LINKER
# ====================================
nlp, LINKER = load_scispacy_model()

# ====================================
# EMBEDDING MODEL & THRESHOLD
# ====================================
MODEL = load_embedding_model()
MATCH_THRESHOLD = 0.9125

# ====================================
# Cache (runtime + disk) + STOP_WORDS
# ====================================
ENTITY_CACHE: dict[str, dict] = {}
ENTITY_RULE_CACHE: dict[str, dict] = {}

INPUT_JSON = "test_results_rate_eval_task2.json"
ENTITY_JSON      = "entity.json"
ENTITY_RULE_JSON = "entity_rule.json"

# ====================================
# Thread Safety
# ====================================
ENTITY_CACHE_LOCK      = threading.Lock()
ENTITY_RULE_CACHE_LOCK = threading.Lock()
SAVE_LOCK              = threading.Lock()

MAX_WORKERS   = 8 
SAVE_INTERVAL = 1000

STOP_WORDS = {
    "of", "the", "a", "an", "in", "on", "at", "to", "for",
    "with", "by", "from", "and", "or", "is", "are", "was",
    "were", "be", "been", "into", "within", "without", "as"
}

# ====================================
# Cache Functions
# ====================================
def load_caches():
    global ENTITY_CACHE, ENTITY_RULE_CACHE
    if Path(ENTITY_JSON).exists():
        with open(ENTITY_JSON, "r", encoding="utf-8") as f:
            ENTITY_CACHE = json.load(f)
        print(f"{ENTITY_JSON} loaded ({len(ENTITY_CACHE)} records)")
    if Path(ENTITY_RULE_JSON).exists():
        with open(ENTITY_RULE_JSON, "r", encoding="utf-8") as f:
            ENTITY_RULE_CACHE = json.load(f)
        print(f"{ENTITY_RULE_JSON} loaded ({len(ENTITY_RULE_CACHE)} records)")

def save_caches():
    """Thread-safe disk write."""
    with SAVE_LOCK:
        with ENTITY_CACHE_LOCK:
            snapshot_entity = dict(ENTITY_CACHE)
        with ENTITY_RULE_CACHE_LOCK:
            snapshot_rule = dict(ENTITY_RULE_CACHE)

        with open(ENTITY_JSON, "w", encoding="utf-8") as f:
            json.dump(snapshot_entity, f, indent=2, ensure_ascii=False)
        with open(ENTITY_RULE_JSON, "w", encoding="utf-8") as f:
            json.dump(snapshot_rule, f, indent=2, ensure_ascii=False)

# ====================================
# HELPERS
# ====================================
def remove_stopwords(text: str) -> str:
    tokens = text.lower().split()
    filtered = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(filtered)

def make_result_dict(term: str, res: dict | None, split_type: str) -> dict:
    if res:
        return {
            "term": term,
            "found": True,
            "source": res["source"],
            "id": res["id"],
            "preferred_name": res["name"],
            "vocabulary": res["vocabulary"],
            "semantic_types": res["semantic_types"],
            "split_type": split_type
        }
    return {
        "term": term,
        "found": False,
        "source": None,
        "id": None,
        "preferred_name": None,
        "vocabulary": None,
        "semantic_types": [],
        "split_type": split_type
    }

# ====================================
# SCISPACY LOGIC (WITH EMBEDDING CHECK)
# ====================================
def resolve_scispacy(term: str) -> dict | None:
    doc = nlp(term)
    
    if not doc.ents:
        return None
    
    best_match_cui = None
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
    
    term_embedding = MODEL.encode(term, convert_to_tensor=True)
    matched_embedding = MODEL.encode(matched_name, convert_to_tensor=True)
    similarity = util.cos_sim(term_embedding, matched_embedding).cpu().numpy()[0][0]
    
    # If similarity is below the threshold, treat it as an irrelevant match and reject.
    if similarity < MATCH_THRESHOLD:
        return None
    # --------------------------------------------------------------
    
    return {
        "source": "UMLS - Scispacy",
        "id": concept.concept_id,
        "name": matched_name,
        "vocabulary": "UMLS", 
        "semantic_types": concept.types,
    }

# =======================================
# Unified Resolver (thread-safe cache)
# =======================================
def resolve_term_cached(term: str, split_type: str) -> dict:
    key = term.lower().strip()

    if key in ENTITY_CACHE:
        return ENTITY_CACHE[key]

    result = resolve_scispacy(term)
    result = make_result_dict(term, result, split_type)

    with ENTITY_CACHE_LOCK:
        if key not in ENTITY_CACHE:
            ENTITY_CACHE[key] = result

    return result

# =======================================
# TOKENIZATION LOGIC
# =======================================
def get_partitions(words: list[str]) -> list[list[str]]:
    if not words:
        return []
    
    # Restriction for large entities -> 2^(n-1) combinations
    if len(words) > 7:
        return [words]
    
    partitions = [[words[0]]]
    for word in words[1:]:
        new_partitions = []
        for p in partitions:
            p1 = p.copy(); p1[-1] += " " + word
            p2 = p.copy(); p2.append(word)
            new_partitions.extend([p1, p2])
        partitions = new_partitions
    return partitions

# =======================================
# Phrase resolver (thread-safe)
# =======================================
def resolve_phrase_combinations(phrase: str, split_type: str) -> dict:
    key = phrase.lower().strip()

    if key in ENTITY_RULE_CACHE:
        return ENTITY_RULE_CACHE[key]

    tokens = key.split()
    all_partitions = get_partitions(tokens)

    unique_tokens = {token for partition in all_partitions for token in partition}
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

# =======================================
# Per-entity worker (thread worker)
# =======================================
def process_single_entity(entry: dict) -> str | None:
    name = entry.get("name", "").strip().lower()
    field_name = entry.get("field_name", "unknown")

    if not name or field_name == "measurement":
        return None

    clean_name = remove_stopwords(name)
    if not clean_name:
        return None

    resolve_phrase_combinations(clean_name, split_type="test")
    return clean_name

# =============
# Main pipeline
# =============
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
    print(f"\nCompleted. Total {processed_count} entities has been processed.")
    print(f"Entity cache: {len(ENTITY_CACHE)} | Rule cache: {len(ENTITY_RULE_CACHE)}")

if __name__ == "__main__":
    process_unique_entities(INPUT_JSON)