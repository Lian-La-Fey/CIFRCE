import json
import re

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple, Dict

from dataclasses import dataclass, field as dc_field
from tqdm import tqdm
from functools import lru_cache

# ================================
# CONFIGURATION
# ================================

INPUT_JSON = "test_results_rate_eval_task2.json"
ENTITY_JSON = "entity.json"
ENTITY_RULE_JSON = "entity_rule.json"
UMLS_CUI_SYNONYMS_JSON = "umls_cui_synonyms.json"
EMBED_SYNONYMS_JSON = "embedding_synonyms.json"
ABBREV_JSON = "medical_abbreviations_dictionary_normalized.json"

FIELDS = ["observation", "location", "degree", "trend"]
OUTPUT_FILENAME = "detailed_evaluation_results.json"
NUM_WORKERS = 2

STOP_WORDS = {
    "of", "the", "a", "an", "in", "on", "at", "to", "for",
    "with", "by", "from", "and", "or", "is", "are", "was",
    "were", "be", "been", "into", "within", "without", "as"
}

# Caches
# --------------------------------

INPUT_DATA: dict = {}
ENTITY_CACHE: dict = {}
ENTITY_RULE: dict = {}
UMLS_CUI_SYNONYMS: dict = {}
EMBED_SYNONYMS: dict = {}
ABBREV: dict = {}

# Data Classes
# --------------------------------

@dataclass
class Entity:
    eid: str
    value: str
    field: str
    status: str
    partitions: List[List[str]] = dc_field(default_factory=list)
    is_matched: bool = False
    match_type: str = ""

@dataclass
class RelationGroup:
    rid: str
    anchor: Entity
    related: List[Entity]
    
    @property
    def all_entities(self) -> List[Entity]:
        return [self.anchor] + self.related

@dataclass
class PoolToken:
    value: str
    field: str
    status: str
    weight: float = 1.0
    cui: Optional[str] = None
    siblings: Tuple[str, ...] = tuple()

class FastPool:
    """O(1) lookup index for relation group pools."""
    def __init__(self, tokens: List[PoolToken]):
        self.by_val = {}
        self.by_cui = {}
        for t in tokens:
            self.by_val.setdefault(t.value, []).append(t)
            if t.cui:
                self.by_cui.setdefault(t.cui, []).append(t)


# ================================
# HELPERS
# ================================

def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
    
def load_all():
    global INPUT_DATA, ENTITY_CACHE, ENTITY_RULE, UMLS_CUI_SYNONYMS, EMBED_SYNONYMS, ABBREV
    
    INPUT_DATA = load_json(INPUT_JSON)
    ENTITY_CACHE = load_json(ENTITY_JSON)
    ENTITY_RULE = load_json(ENTITY_RULE_JSON)
    UMLS_CUI_SYNONYMS = load_json(UMLS_CUI_SYNONYMS_JSON)
    EMBED_SYNONYMS = load_json(EMBED_SYNONYMS_JSON)
    ABBREV = load_json(ABBREV_JSON)

@lru_cache(maxsize=32_768)    
def parse_entity(text: str):
    text = text.strip()
    match = re.match(r'^(.+?)\s*\(([\w]+):([\w]+)\)\s*$', text)
    
    if match:
        value = match.group(1).strip().lower()
        field = match.group(2).strip().lower()
        status = match.group(3).strip().lower()
        return value, field, status
    
    print(text)
    return None, None, None

@lru_cache(maxsize=32_768)
def remove_stopwords(text: str) -> str:
    tokens = text.lower().split()
    return " ".join(t for t in tokens if t not in STOP_WORDS)

def collect_field_entities(groups: List[RelationGroup], field: str) -> List[Tuple[RelationGroup, Entity]]:
    out = []
    for g in groups:
        for e in g.all_entities:
            if e.field == field:
                out.append((g, e))
    return out

# ================================
# KNOWLEDGE BASE HELPERS
# ================================

@lru_cache(maxsize=8_192)
def expand_abbreviation(term: str) -> List[str]:
    low = term.strip().lower()
    if low in ABBREV:
        return [expansion.lower() for expansion in ABBREV[low]]
    return []

@lru_cache(maxsize=32_768)
def get_embedding_synonyms_with_sim(term: str) -> Dict[str, float]:
    """Returns similarity (sim) scores, not just the values."""
    term = term.lower()
    if term in EMBED_SYNONYMS:
        return {item["value"]: item["sim"] for item in EMBED_SYNONYMS[term] if item["sim"] > 0.9}
    return {}

@lru_cache(maxsize=32_768)
def get_term_cui(term: str) -> Optional[str]:
    term = term.lower()
    if term in UMLS_CUI_SYNONYMS:
        return UMLS_CUI_SYNONYMS[term]["cui"]
    if term in ENTITY_CACHE and ENTITY_CACHE[term].get("found"):
        return ENTITY_CACHE[term]["id"]
    return None

@lru_cache(maxsize=32_768)
def get_cui_synonyms(term: str) -> List[str]:
    return UMLS_CUI_SYNONYMS.get(term.lower(), {}).get("synonyms", [])

def create_relation_group(item: dict, rg_id: str) -> RelationGroup:
    anchor_val, anchor_f, anchor_s = parse_entity(item["anchor"])
    anchor_entity = Entity(eid=f"{rg_id}_anchor", value=anchor_val, field=anchor_f, status=anchor_s)
    
    related_entities = []
    for idx, rel_str in enumerate(item.get("related", [])):
        r_val, r_f, r_s = parse_entity(rel_str)
        related_entities.append(
            Entity(eid=f"{rg_id}_rel_{idx}", value=r_val, field=r_f, status=r_s)
        )
        
    return RelationGroup(rid=rg_id, anchor=anchor_entity, related=related_entities)

# ================================
# PARTITION & POOL HELPERS
# ================================
 
@lru_cache(maxsize=32_768)
def get_entity_partitions(term: str) -> List[List[str]]:
    clean = remove_stopwords(term).lower()
    if clean in ENTITY_RULE:
        return ENTITY_RULE[clean]["rules"]
    return [[term]]

def build_group_pool(group: RelationGroup) -> FastPool:
    pool_list = []
    seen_weights = {} 
    
    for e in group.all_entities:
        clean = remove_stopwords(e.value).lower()
        
        expansions = {clean: 1.0}
        
        for abv in expand_abbreviation(clean):
            expansions[abv] = 1.0
            
        for csyn in get_cui_synonyms(clean):
            if expansions.get(csyn, 0.0) < 0.9:
                expansions[csyn] = 0.9
                
        for esyn, sim in get_embedding_synonyms_with_sim(clean).items():
            if expansions.get(esyn, 0.0) < sim:
                expansions[esyn] = sim
                
        for form, weight in expansions.items():
            if not form:
                continue
            parts = get_entity_partitions(form)
            for p_list in parts:
                for i, token in enumerate(p_list):
                    tok_lower = token.lower()
                    tok_field = e.field if tok_lower == form.lower() else ""
                    
                    siblings = tuple(t.lower() for j, t in enumerate(p_list) if j != i)
                    
                    pt_tuple = (tok_lower, tok_field, e.status, siblings)
                    
                    if pt_tuple not in seen_weights or seen_weights[pt_tuple] < weight:
                        seen_weights[pt_tuple] = weight

                    # ── NEW BLOCK ─────────────────────────────────────────────
                    # A partition token can itself be an abbreviation.
                    # Example: ["acute", "mi", "pain"] → "mi" → "myocardial infarction"
                    # In that case, we also add the expanded tokens to the pool.
                    for abv_exp in expand_abbreviation(tok_lower):
                        abv_parts = get_entity_partitions(abv_exp)
                        for abv_p_list in abv_parts:
                            for k, abv_tok in enumerate(abv_p_list):
                                abv_tok_lower = abv_tok.lower()
                                # Siblings: outer partition siblings +
                                #           expansion's internal siblings
                                abv_siblings = siblings + tuple(
                                    t.lower() for m, t in enumerate(abv_p_list) if m != k
                                )
                                abv_tuple = (abv_tok_lower, "", e.status, abv_siblings)
                                if abv_tuple not in seen_weights or seen_weights[abv_tuple] < weight:
                                    seen_weights[abv_tuple] = weight
                    # ── END NEW BLOCK ─────────────────────────────────────────
                        
    for (val, field, status, siblings), weight in seen_weights.items():
        pool_list.append(PoolToken(
            value=val, field=field, status=status, 
            weight=weight, cui=get_term_cui(val), siblings=siblings
        ))
        
    return FastPool(pool_list)

def check_field_status_match(field_1: str, status_1: str, field_2: str, status_2: str, ignore_status: bool = False) -> bool:
    """
    1. If both fields are non-empty and different -> always False (regardless of ignore_status).
    2. If ignore_status = True (and rule 1 does not apply) -> always True.
    3. If ignore_status = False:
        a. For fields other than "location" (including empty, "observation", "degree", "trend") -> True only if status_1 == status_2.
        b. For "location": 
            If exactly one field is empty -> True (no status check).
            If both fields are "location" -> True only if status_1 == status_2.
    
    Args:
        field_1 (str)   : (observation | location | degree | trend)
        status_1 (str)  : (present | absent | uncertain)
        field_2 (str)   : (observation | location | degree | trend)
        status_2 (str)  : (present | absent | uncertain)
        ignore_status   : (True | False)

    Returns:
        bool: (True | False)
    """
    
    if field_1 != "" and field_2 != "" and field_1 != field_2:
        return False
        
    if ignore_status:
        return True

    eff_field = field_1 if field_1 != "" else (field_2 if field_2 != "" else "")
    
    if eff_field == "location":
        if field_1 == "" or field_2 == "":
            return True
        else:
            return status_1 == status_2
    else:
        return status_1 == status_2

def verify_siblings(siblings: Tuple[str, ...], source_pool: FastPool) -> bool:
    """Called only when truly needed."""
    if not siblings:
        return True
        
    for sib in siblings:
        if sib in source_pool.by_val:
            continue
            
        cui = get_term_cui(sib)
        if cui and cui in source_pool.by_cui:
            continue
            
    # Embedding check is the most expensive, so keep it last.
        found = False
        # get_embedding_synonyms_with_sim zaten lru_cache'li
        for esyn in get_embedding_synonyms_with_sim(sib):
            if esyn in source_pool.by_val:
                found = True
                break
        if not found:
            return False
    return True

def get_token_match_score(target_val: str, target_field: str, target_status: str, fast_pool: FastPool, source_pool: FastPool, ignore_status: bool = False) -> float:
    token_lower = target_val.lower()
    best_score = 0.0

    def is_valid(p_token: PoolToken) -> bool:
    # Skip if field mismatch exists.
        if target_field and p_token.field and target_field != p_token.field: 
            return False
        
        if ignore_status: 
            return True
            
    # Location-specific rule and general status check.
        eff_field = target_field or p_token.field or ""
        if eff_field == "location" and (not target_field or not p_token.field): 
            return True
        return target_status == p_token.status

    # 1. Exact Match Lookups
    if token_lower in fast_pool.by_val:
        for p_token in fast_pool.by_val[token_lower]:
            if not is_valid(p_token): 
                continue
            # Önce skor hesapla
            score = p_token.weight if target_status == p_token.status else p_token.weight * 0.95
            # Only run heavy checks (siblings) if the score can be the best.
            if score > best_score:
                if not p_token.siblings or verify_siblings(p_token.siblings, source_pool):
                    best_score = score

    if best_score == 1.0: 
        return 1.0

    # 2. CUI Match Lookups
    cui1 = get_term_cui(token_lower)
    if cui1 and cui1 in fast_pool.by_cui:
        for p_token in fast_pool.by_cui[cui1]:
            if not is_valid(p_token): 
                continue
            score = 0.9 * p_token.weight
            if target_status != p_token.status: 
                score *= 0.95
            if score > best_score:
                if not p_token.siblings or verify_siblings(p_token.siblings, source_pool):
                    best_score = score

    if best_score >= 0.9:
        return best_score  # Optional: if CUI match is very high, you may skip embedding.

    # 3. Embedding Synonym Lookups
    embed_syns_target = get_embedding_synonyms_with_sim(token_lower)
    for esyn, sim in embed_syns_target.items():
        if esyn not in fast_pool.by_val:
            continue
            
        for p_token in fast_pool.by_val[esyn]:
            if not is_valid(p_token): 
                continue
            score = sim * p_token.weight
            if target_status != p_token.status: 
                score *= 0.95
            if score > best_score:
                if not p_token.siblings or verify_siblings(p_token.siblings, source_pool):
                    best_score = score
    
    return best_score

# ================================
# FLEXIBLE MATCHING
# ================================

def get_entity_pool_score(target_val: str, target_field: str, target_status: str, pool: FastPool, source_pool: FastPool, ignore_status: bool = False) -> float:
    clean = remove_stopwords(target_val).lower()
    if not clean: return 0.0
        
    partitions = get_entity_partitions(clean)
    best_part_score = 0.0
    
    for parts in partitions:
        part_score_sum = 0.0
        all_matched = True
        
        for p in parts:
            q_field = target_field if p == clean else ""
            
            # source_pool is passed here
            tok_score = get_token_match_score(p, q_field, target_status, pool, source_pool, ignore_status)
            if tok_score == 0.0:
                all_matched = False
                break
            part_score_sum += tok_score
        
        if all_matched:
            avg_score = part_score_sum / len(parts)
            if avg_score > best_part_score:
                best_part_score = avg_score
                
    return best_part_score

def flexible_match(ent: Entity, pool2: FastPool, source_pool: FastPool, ignore_status: bool = False) -> Tuple[bool, str, float]:
    best_score = 0.0
    best_match_type = ""

    score = get_entity_pool_score(ent.value, ent.field, ent.status, pool2, source_pool, ignore_status)
    if score > best_score:
        best_score = score
        best_match_type = "flexible_direct"
    if best_score == 1.0: return True, best_match_type, best_score

    for abv in expand_abbreviation(ent.value):
        score = get_entity_pool_score(abv, ent.field, ent.status, pool2, source_pool, ignore_status)
        if score > best_score:
            best_score = score
            best_match_type = "flexible_expanded_abbrev"
    if best_score == 1.0: return True, best_match_type, best_score

    for csyn in get_cui_synonyms(ent.value):
        score = get_entity_pool_score(csyn, ent.field, ent.status, pool2, source_pool, ignore_status)
        score *= 0.9  
        if score > best_score:
            best_score = score
            best_match_type = "flexible_expanded_cui"

    for esyn, sim in get_embedding_synonyms_with_sim(ent.value).items():
        score = get_entity_pool_score(esyn, ent.field, ent.status, pool2, source_pool, ignore_status)
        score *= sim  
        if score > best_score:
            best_score = score
            best_match_type = "flexible_expanded_embed"

    is_matched = best_score > 0.0
    return is_matched, best_match_type, best_score

def is_groups_anchor_matched(group1: RelationGroup, pool2: FastPool, source_pool: FastPool, ignore_status: bool = True) -> bool:
    is_m, _, _ = flexible_match(group1.anchor, pool2, source_pool, ignore_status=ignore_status)
    return is_m

def _get_flexible_score(group1: RelationGroup, pool2: FastPool, source_pool: FastPool, ignore_status: bool = True) -> float:
    total_score = 0.0
    for e1 in group1.all_entities:
        is_m, _, score = flexible_match(e1, pool2, source_pool, ignore_status=ignore_status)
        if is_m:
            total_score += score
    return total_score
 
# ================================
# MAIN MATCHING PIPELINE
# ================================
 
def match_entities(
    gt_groups:   List[RelationGroup],
    pred_groups: List[RelationGroup],
    field:       str,
    gt_pool_cache,
    pred_pool_cache
) -> Tuple[List[Entity], List[Entity], List[Entity], List[Entity]]:
    
    gt_pairs   = collect_field_entities(gt_groups,   field)
    pred_pairs = collect_field_entities(pred_groups, field)
    
    gt_matched, gt_unmatched = [], []
    pred_matched, pred_unmatched = [], []

    def _collect_candidates(
        prior_group: RelationGroup,
        ent: Entity,
        target_groups: List[RelationGroup],
        source_pool: FastPool,
        target_pool_cache: dict,
        ignore_status: bool = True,
    ) -> List[dict]:
        candidates = []
        for target_group in target_groups:
            pool2 = target_pool_cache[target_group.rid]

            if not ent.eid.endswith("_anchor"):
                if not is_groups_anchor_matched(prior_group, pool2, source_pool, ignore_status=True):
                    continue

            is_matched_soft, _, _ = flexible_match(ent, pool2, source_pool, ignore_status=ignore_status)
            if is_matched_soft:
                candidates.append({
                    "prior_group": prior_group,
                    "secondary_group": target_group,
                    "target_pool": pool2,
                })
        return candidates

    def _pick_best_candidate(candidates: List[dict], source_pool: FastPool) -> Optional[dict]:
        if not candidates:
            return None
        for c in candidates:
            c["score"] = _get_flexible_score(c["prior_group"], c["target_pool"], source_pool, ignore_status=True)
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[0]

    # ===============================
    # PASS 1  –  GT -> PRED
    # ===============================
    for gt_group, gt_ent in gt_pairs:
        source_pool = gt_pool_cache[gt_group.rid]  # Query pool
        candidates = _collect_candidates(
            prior_group=gt_group,
            ent=gt_ent,
            target_groups=pred_groups,
            source_pool=source_pool,
            target_pool_cache=pred_pool_cache,
            ignore_status=True,
        )

        best = _pick_best_candidate(candidates, source_pool)
        if best is None:
            gt_unmatched.append(gt_ent)
            continue

        is_matched_hard, match_type_hard, _ = flexible_match(
            gt_ent,
            best["target_pool"],
            source_pool,
            ignore_status=False,
        )

        if is_matched_hard:
            gt_ent.is_matched = True
            gt_ent.match_type = match_type_hard
            gt_matched.append(gt_ent)
        else:
            gt_unmatched.append(gt_ent)

    # ===============================
    # PASS 2  –  PRED -> GT
    # ===============================
    for pred_group, pred_ent in pred_pairs:
        source_pool = pred_pool_cache[pred_group.rid]  # Query pool
        candidates = _collect_candidates(
            prior_group=pred_group,
            ent=pred_ent,
            target_groups=gt_groups,
            source_pool=source_pool,
            target_pool_cache=gt_pool_cache,
            ignore_status=True,
        )

        best = _pick_best_candidate(candidates, source_pool)
        if best is None:
            pred_unmatched.append(pred_ent)
            continue

        is_matched_hard, match_type_hard, _ = flexible_match(
            pred_ent,
            best["target_pool"],
            source_pool,
            ignore_status=False,
        )

        if is_matched_hard:
            pred_ent.is_matched = True
            pred_ent.match_type = match_type_hard
            pred_matched.append(pred_ent)
        else:
            pred_unmatched.append(pred_ent)

    return gt_matched, gt_unmatched, pred_matched, pred_unmatched
 
# ================================
# EXPORT HELPERS & WORKER
# ================================

def init_worker():
    load_all()

def entity_to_dict(ent: Entity) -> dict:
    """Convert an Entity dataclass to a detailed dictionary."""
    return {
        "eid": ent.eid,
        "value": ent.value,
        "field": ent.field,
        "status": ent.status,
        "is_matched": ent.is_matched,
        "match_type": ent.match_type
    }

def process_single_report(report: dict) -> dict:
    """Worker function that runs matching logic for a single report."""
    
    gt_groups = [create_relation_group(item, f"gt_{idx}") for idx, item in enumerate(report.get("gt_entities", []))]
    pred_groups = [create_relation_group(item, f"pred_{idx}") for idx, item in enumerate(report.get("pred_entities", []))]
    
    gt_pool_cache = {g.rid: build_group_pool(g) for g in gt_groups}
    pred_pool_cache = {g.rid: build_group_pool(g) for g in pred_groups}
    
    evaluation_summary = {}

    for field in FIELDS:
        gt_m, gt_u, pred_m, pred_u = match_entities(
            gt_groups, pred_groups, field=field,
            gt_pool_cache=gt_pool_cache,
            pred_pool_cache=pred_pool_cache
        )
        
        evaluation_summary[field] = {
            "ground_truth": {
                "matched": [entity_to_dict(e) for e in gt_m],
                "unmatched": [entity_to_dict(e) for e in gt_u]
            },
            "prediction": {
                "matched": [entity_to_dict(e) for e in pred_m],
                "unmatched": [entity_to_dict(e) for e in pred_u]
            }
        }
    
    report["evaluation_results"] = evaluation_summary
    
    return report

# ================================
# MAIN
# ================================

def main():
    load_all()
    
    detailed_results = []
    
    print(
        f"Processing {len(INPUT_DATA)} records... (Workers: {NUM_WORKERS})"
    )
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker) as executor:
        futures = {executor.submit(process_single_report, report): report for report in INPUT_DATA}
        
    # Use as_completed to show completed tasks in the progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Results"):
            try:
                result = future.result()
                detailed_results.append(result)
            except Exception as exc:
                print(f"An error occurred while processing a report: {exc}")

    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as out_file:
        json.dump(detailed_results, out_file, ensure_ascii=False, indent=2)
        
    print(
        f"\nDone! Detailed results saved to '{OUTPUT_FILENAME}'."
    )
 
if __name__ == "__main__":
    main()