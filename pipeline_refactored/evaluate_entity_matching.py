"""
Step 11 — Flexible Entity-Match Evaluation (Fast, Multi-Process)
=================================================================
For every report in *test_results_rate_eval_task2.json* this script
bi-directionally matches ground-truth (GT) vs. predicted (PRED) entity
relation groups across four semantic fields:

    observation · location · degree · trend

Matching is *flexible*: it expands each entity via abbreviation dictionaries,
UMLS CUI synonyms, and BioLORD embedding synonyms before scoring.  A
two-pass approach (soft anchor check → hard status check) avoids
over-penalising partial matches.

Outputs:
    detailed_evaluation_results.json  —  per-report entity-level match details

Usage:
    python 11_entity_match_evaluation_flexible_score_fast.py
"""

import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field as dc_field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# ================================
# CONFIGURATION
# ================================

INPUT_JSON            = "test_results_rate_eval_task2.json"
ENTITY_JSON           = "entity.json"
ENTITY_RULE_JSON      = "entity_rule.json"
UMLS_CUI_SYNONYMS_JSON = "umls_cui_synonyms.json"
EMBED_SYNONYMS_JSON   = "embedding_synonyms.json"
ABBREV_JSON           = "medical_abbreviations_dictionary_normalized.json"

OUTPUT_JSON           = "detailed_evaluation_results.json"
NUM_WORKERS           = 2

SEMANTIC_FIELDS       = ["observation", "location", "degree", "trend"]

STOP_WORDS = {
    "of", "the", "a", "an", "in", "on", "at", "to", "for",
    "with", "by", "from", "and", "or", "is", "are", "was",
    "were", "be", "been", "into", "within", "without", "as",
}


# ================================
# GLOBAL CACHES  (populated by load_all)
# ================================

INPUT_DATA: list       = []
ENTITY_CACHE: dict     = {}
ENTITY_RULE: dict      = {}
UMLS_CUI_SYNONYMS: dict = {}
EMBED_SYNONYMS: dict   = {}
ABBREV: dict           = {}


# ================================
# DATA CLASSES
# ================================

@dataclass
class Entity:
    """A single medical entity with its semantic field and presence status."""
    eid:        str
    value:      str
    field:      str
    status:     str
    partitions: List[List[str]] = dc_field(default_factory=list)
    is_matched: bool = False
    match_type: str  = ""


@dataclass
class RelationGroup:
    """An anchor entity plus its semantically related entities."""
    rid:     str
    anchor:  Entity
    related: List[Entity]

    @property
    def all_entities(self) -> List[Entity]:
        return [self.anchor] + self.related


@dataclass
class PoolToken:
    """A single lookup token inside a :class:`FastPool`."""
    value:    str
    field:    str
    status:   str
    weight:   float = 1.0
    cui:      Optional[str] = None
    siblings: Tuple[str, ...] = ()


class FastPool:
    """O(1) lookup index over the relation-group token pool.

    Provides two secondary indices:
      * ``by_val``  — tokens keyed by surface form
      * ``by_cui``  — tokens keyed by UMLS CUI
    """

    def __init__(self, tokens: List[PoolToken]) -> None:
        self.by_val: Dict[str, List[PoolToken]] = {}
        self.by_cui: Dict[str, List[PoolToken]] = {}
        for t in tokens:
            self.by_val.setdefault(t.value, []).append(t)
            if t.cui:
                self.by_cui.setdefault(t.cui, []).append(t)


# ================================
# I/O HELPERS
# ================================

def _load_json(path: str):
    """Load and return the contents of a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all() -> None:
    """Populate all global caches from disk.  Must be called before any matching."""
    global INPUT_DATA, ENTITY_CACHE, ENTITY_RULE, UMLS_CUI_SYNONYMS, EMBED_SYNONYMS, ABBREV
    INPUT_DATA        = _load_json(INPUT_JSON)
    ENTITY_CACHE      = _load_json(ENTITY_JSON)
    ENTITY_RULE       = _load_json(ENTITY_RULE_JSON)
    UMLS_CUI_SYNONYMS = _load_json(UMLS_CUI_SYNONYMS_JSON)
    EMBED_SYNONYMS    = _load_json(EMBED_SYNONYMS_JSON)
    ABBREV            = _load_json(ABBREV_JSON)


# ================================
# TEXT HELPERS
# ================================

@lru_cache(maxsize=32_768)
def parse_entity(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse a raw entity string of the form::

        value (field:status)

    Returns ``(value, field, status)`` — all lower-cased — or
    ``(None, None, None)`` when the pattern does not match.
    """
    text  = text.strip()
    match = re.match(r'^(.+?)\s*\(([\w]+):([\w]+)\)\s*$', text)
    if match:
        return (
            match.group(1).strip().lower(),
            match.group(2).strip().lower(),
            match.group(3).strip().lower(),
        )
    print(f"[parse_entity] No match for: {text}")
    return None, None, None


@lru_cache(maxsize=32_768)
def remove_stopwords(text: str) -> str:
    """Return *text* with stop-words removed (case-insensitive)."""
    tokens = text.lower().split()
    return " ".join(t for t in tokens if t not in STOP_WORDS)


# ================================
# KNOWLEDGE-BASE HELPERS
# ================================

@lru_cache(maxsize=8_192)
def expand_abbreviation(term: str) -> List[str]:
    """Return all known expansions for *term* from the abbreviation dictionary."""
    low = term.strip().lower()
    return [exp.lower() for exp in ABBREV.get(low, [])]


@lru_cache(maxsize=32_768)
def get_embedding_synonyms_with_sim(term: str) -> Dict[str, float]:
    """
    Return a dict of ``{synonym: similarity_score}`` for *term* from the
    pre-computed BioLORD embedding synonyms (only entries with sim > 0.9).
    """
    term = term.lower()
    if term in EMBED_SYNONYMS:
        return {item["value"]: item["sim"] for item in EMBED_SYNONYMS[term] if item["sim"] > 0.9}
    return {}


@lru_cache(maxsize=32_768)
def get_term_cui(term: str) -> Optional[str]:
    """Return the UMLS CUI for *term*, checking UMLS synonyms then the entity cache."""
    term = term.lower()
    if term in UMLS_CUI_SYNONYMS:
        return UMLS_CUI_SYNONYMS[term]["cui"]
    if term in ENTITY_CACHE and ENTITY_CACHE[term].get("found"):
        return ENTITY_CACHE[term]["id"]
    return None


@lru_cache(maxsize=32_768)
def get_cui_synonyms(term: str) -> List[str]:
    """Return UMLS CUI-level synonyms for *term*."""
    return UMLS_CUI_SYNONYMS.get(term.lower(), {}).get("synonyms", [])


# ================================
# RELATION GROUP CONSTRUCTION
# ================================

def _make_entity(raw: str, eid: str) -> Entity:
    """Parse *raw* and return an :class:`Entity` with the given *eid*."""
    value, field, status = parse_entity(raw)
    return Entity(eid=eid, value=value, field=field, status=status)


def create_relation_group(item: dict, rg_id: str) -> RelationGroup:
    """
    Build a :class:`RelationGroup` from a raw dict entry.

    Args:
        item:  Dict with ``"anchor"`` (str) and ``"related"`` (list[str]).
        rg_id: Unique ID prefix for entity IDs.
    """
    anchor = _make_entity(item["anchor"], f"{rg_id}_anchor")
    related = [
        _make_entity(rel_str, f"{rg_id}_rel_{idx}")
        for idx, rel_str in enumerate(item.get("related", []))
    ]
    return RelationGroup(rid=rg_id, anchor=anchor, related=related)


# ================================
# PARTITION & POOL HELPERS
# ================================

@lru_cache(maxsize=32_768)
def get_entity_partitions(term: str) -> List[List[str]]:
    """
    Look up pre-computed partition rules for *term* from ENTITY_RULE, or
    fall back to a single-partition list containing the full term.
    """
    clean = remove_stopwords(term).lower()
    if clean in ENTITY_RULE:
        return ENTITY_RULE[clean]["rules"]
    return [[term]]


def _add_expansion_tokens(
    form: str,
    weight: float,
    status: str,
    outer_siblings: Tuple[str, ...],
    seen_weights: dict,
) -> None:
    """
    Expand *form* into partition tokens (and their abbreviation expansions)
    and register them in *seen_weights* with the given *weight*.

    This is a helper extracted from :func:`build_group_pool` to keep the main
    function readable.
    """
    parts_list = get_entity_partitions(form)

    for p_list in parts_list:
        for i, token in enumerate(p_list):
            tok_lower  = token.lower()
            tok_field  = form.lower() if tok_lower == form.lower() else ""
            siblings   = tuple(t.lower() for j, t in enumerate(p_list) if j != i)

            pt_tuple = (tok_lower, tok_field, status, siblings)
            if seen_weights.get(pt_tuple, -1) < weight:
                seen_weights[pt_tuple] = weight

            # Also expand abbreviations within the partition token itself
            for abv_exp in expand_abbreviation(tok_lower):
                abv_parts_list = get_entity_partitions(abv_exp)
                for abv_p_list in abv_parts_list:
                    for k, abv_tok in enumerate(abv_p_list):
                        abv_tok_lower = abv_tok.lower()
                        abv_siblings  = outer_siblings + tuple(
                            t.lower() for m, t in enumerate(abv_p_list) if m != k
                        )
                        abv_tuple = (abv_tok_lower, "", status, abv_siblings)
                        if seen_weights.get(abv_tuple, -1) < weight:
                            seen_weights[abv_tuple] = weight


def build_group_pool(group: RelationGroup) -> FastPool:
    """
    Build a :class:`FastPool` index for all tokens in *group*.

    Each entity is expanded via abbreviations, UMLS CUI synonyms, and
    BioLORD embedding synonyms.  Weights reflect expansion confidence:
    direct = 1.0, CUI = 0.9, embedding = cosine similarity score.
    """
    seen_weights: dict = {}

    for e in group.all_entities:
        clean = remove_stopwords(e.value).lower()

        expansions: Dict[str, float] = {clean: 1.0}

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
            outer_siblings: Tuple[str, ...] = ()
            _add_expansion_tokens(form, weight, e.status, outer_siblings, seen_weights)

    pool_list = [
        PoolToken(
            value=val, field=field, status=status,
            weight=weight, cui=get_term_cui(val), siblings=siblings,
        )
        for (val, field, status, siblings), weight in seen_weights.items()
    ]
    return FastPool(pool_list)


# ================================
# FIELD / STATUS MATCHING RULES
# ================================

def check_field_status_match(
    field_1: str, status_1: str,
    field_2: str, status_2: str,
    ignore_status: bool = False,
) -> bool:
    """
    Determine whether two (field, status) pairs are compatible.

    Rules:
      1. If both fields are non-empty and different → always False.
      2. If *ignore_status* is True (and rule 1 does not apply) → always True.
      3. Otherwise:
           - For non-location fields: True only if ``status_1 == status_2``.
           - For "location": if exactly one field is empty → True (no status check);
             if both are "location" → True only if statuses match.

    Args:
        field_1, status_1: Field and status of the first entity.
        field_2, status_2: Field and status of the second entity.
        ignore_status:     When True skip the status check (soft pass).

    Returns:
        True if the pair is compatible, False otherwise.
    """
    if field_1 and field_2 and field_1 != field_2:
        return False
    if ignore_status:
        return True

    eff_field = field_1 or field_2 or ""
    if eff_field == "location":
        if not field_1 or not field_2:
            return True
        return status_1 == status_2

    return status_1 == status_2


# ================================
# SIBLING VERIFICATION
# ================================

def verify_siblings(siblings: Tuple[str, ...], source_pool: FastPool) -> bool:
    """
    Check that every sibling token is reachable in *source_pool* (by value,
    CUI, or embedding synonym).  Called only when a candidate token has
    siblings, avoiding unnecessary overhead.
    """
    if not siblings:
        return True

    for sib in siblings:
        if sib in source_pool.by_val:
            continue

        cui = get_term_cui(sib)
        if cui and cui in source_pool.by_cui:
            continue

        if any(esyn in source_pool.by_val for esyn in get_embedding_synonyms_with_sim(sib)):
            continue

        return False

    return True


# ================================
# TOKEN-LEVEL SCORING
# ================================

def get_token_match_score(
    target_val:    str,
    target_field:  str,
    target_status: str,
    fast_pool:     FastPool,
    source_pool:   FastPool,
    ignore_status: bool = False,
) -> float:
    """
    Score the match between a single query token and a :class:`FastPool`.

    Three lookup strategies are attempted in order of cost:
      1. Exact value match (O(1) via ``by_val``).
      2. CUI-level match (O(1) via ``by_cui``).
      3. Embedding-synonym match (O(|synonyms|)).

    Early-exit at score 1.0 (exact) or ≥ 0.9 (CUI) to avoid unnecessary work.

    Returns:
        Best match score in [0.0, 1.0].
    """
    token_lower = target_val.lower()
    best_score  = 0.0

    def is_valid(p_token: PoolToken) -> bool:
        if target_field and p_token.field and target_field != p_token.field:
            return False
        if ignore_status:
            return True
        eff_field = target_field or p_token.field or ""
        if eff_field == "location" and (not target_field or not p_token.field):
            return True
        return target_status == p_token.status

    # 1. Exact value match
    for p_token in fast_pool.by_val.get(token_lower, []):
        if not is_valid(p_token):
            continue
        score = p_token.weight if target_status == p_token.status else p_token.weight * 0.95
        if score > best_score:
            if not p_token.siblings or verify_siblings(p_token.siblings, source_pool):
                best_score = score

    if best_score == 1.0:
        return 1.0

    # 2. CUI match
    cui1 = get_term_cui(token_lower)
    for p_token in fast_pool.by_cui.get(cui1 or "", []):
        if not is_valid(p_token):
            continue
        score = 0.9 * p_token.weight
        if target_status != p_token.status:
            score *= 0.95
        if score > best_score:
            if not p_token.siblings or verify_siblings(p_token.siblings, source_pool):
                best_score = score

    if best_score >= 0.9:
        return best_score

    # 3. Embedding synonym match
    for esyn, sim in get_embedding_synonyms_with_sim(token_lower).items():
        for p_token in fast_pool.by_val.get(esyn, []):
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
# ENTITY-LEVEL SCORING
# ================================

def get_entity_pool_score(
    target_val:    str,
    target_field:  str,
    target_status: str,
    pool:          FastPool,
    source_pool:   FastPool,
    ignore_status: bool = False,
) -> float:
    """
    Score the match between an entity and a :class:`FastPool` by evaluating
    all possible phrase partitions and returning the best average token score
    (only considering fully matched partitions).
    """
    clean = remove_stopwords(target_val).lower()
    if not clean:
        return 0.0

    partitions     = get_entity_partitions(clean)
    best_part_score = 0.0

    for parts in partitions:
        part_score_sum = 0.0
        all_matched    = True

        for p in parts:
            q_field   = target_field if p == clean else ""
            tok_score = get_token_match_score(
                p, q_field, target_status, pool, source_pool, ignore_status
            )
            if tok_score == 0.0:
                all_matched = False
                break
            part_score_sum += tok_score

        if all_matched:
            avg_score = part_score_sum / len(parts)
            if avg_score > best_part_score:
                best_part_score = avg_score

    return best_part_score


# ================================
# FLEXIBLE MATCHING
# ================================

def flexible_match(
    ent:           Entity,
    pool2:         FastPool,
    source_pool:   FastPool,
    ignore_status: bool = False,
) -> Tuple[bool, str, float]:
    """
    Match *ent* against *pool2* using four strategies in order:

    1. Direct value match.
    2. Abbreviation-expansion match.
    3. UMLS CUI-synonym match (score × 0.9).
    4. BioLORD embedding-synonym match (score × similarity).

    Args:
        ent:           The query entity.
        pool2:         The target pool to match against.
        source_pool:   The query-group pool (used for sibling verification).
        ignore_status: When True, skip the status constraint (soft pass).

    Returns:
        ``(is_matched, match_type, best_score)``
    """
    best_score      = 0.0
    best_match_type = ""

    def _update(score: float, match_type: str) -> None:
        nonlocal best_score, best_match_type
        if score > best_score:
            best_score      = score
            best_match_type = match_type

    # 1. Direct
    _update(
        get_entity_pool_score(ent.value, ent.field, ent.status, pool2, source_pool, ignore_status),
        "flexible_direct",
    )
    if best_score == 1.0:
        return True, best_match_type, best_score

    # 2. Abbreviation expansions
    for abv in expand_abbreviation(ent.value):
        _update(
            get_entity_pool_score(abv, ent.field, ent.status, pool2, source_pool, ignore_status),
            "flexible_expanded_abbrev",
        )
    if best_score == 1.0:
        return True, best_match_type, best_score

    # 3. CUI synonyms
    for csyn in get_cui_synonyms(ent.value):
        score = get_entity_pool_score(
            csyn, ent.field, ent.status, pool2, source_pool, ignore_status
        ) * 0.9
        _update(score, "flexible_expanded_cui")

    # 4. Embedding synonyms
    for esyn, sim in get_embedding_synonyms_with_sim(ent.value).items():
        score = get_entity_pool_score(
            esyn, ent.field, ent.status, pool2, source_pool, ignore_status
        ) * sim
        _update(score, "flexible_expanded_embed")

    return best_score > 0.0, best_match_type, best_score


# ================================
# ANCHOR & GROUP-SCORE HELPERS
# ================================

def is_groups_anchor_matched(
    group1:        RelationGroup,
    pool2:         FastPool,
    source_pool:   FastPool,
    ignore_status: bool = True,
) -> bool:
    """Return True if group1's anchor matches anything in *pool2*."""
    is_m, _, _ = flexible_match(group1.anchor, pool2, source_pool, ignore_status=ignore_status)
    return is_m


def _get_flexible_score(
    group1:        RelationGroup,
    pool2:         FastPool,
    source_pool:   FastPool,
    ignore_status: bool = True,
) -> float:
    """Return the sum of flexible match scores for all entities in *group1*."""
    return sum(
        score
        for e1 in group1.all_entities
        for is_m, _, score in [flexible_match(e1, pool2, source_pool, ignore_status=ignore_status)]
        if is_m
    )


# ================================
# FIELD COLLECTION HELPER
# ================================

def collect_field_entities(
    groups: List[RelationGroup], field: str
) -> List[Tuple[RelationGroup, Entity]]:
    """Return all (group, entity) pairs whose entity field equals *field*."""
    return [
        (g, e)
        for g in groups
        for e in g.all_entities
        if e.field == field
    ]


# ================================
# MAIN MATCHING PIPELINE
# ================================

def _find_best_candidate(
    query_group:   RelationGroup,
    query_ent:     Entity,
    target_groups: List[RelationGroup],
    source_pool_cache: dict,
    target_pool_cache: dict,
) -> Optional[dict]:
    """
    Find the highest-scoring target group for *query_ent* using a soft
    anchor pre-filter followed by group-level scoring.

    Returns the best candidate dict (keys: ``prior_group``, ``secondary_group``,
    ``target_pool``, ``score``), or *None* if no soft match is found.
    """
    source_pool = source_pool_cache[query_group.rid]
    candidates  = []

    for target_group in target_groups:
        target_pool = target_pool_cache[target_group.rid]

        # Non-anchor entities require the group anchor to match softly first
        if not query_ent.eid.endswith("_anchor"):
            if not is_groups_anchor_matched(query_group, target_pool, source_pool, ignore_status=True):
                continue

        is_soft, _, _ = flexible_match(query_ent, target_pool, source_pool, ignore_status=True)
        if is_soft:
            candidates.append({
                "prior_group":     query_group,
                "secondary_group": target_group,
                "target_pool":     target_pool,
            })

    if not candidates:
        return None

    for c in candidates:
        c["score"] = _get_flexible_score(c["prior_group"], c["target_pool"], source_pool, ignore_status=True)

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[0]


def match_entities(
    gt_groups:        List[RelationGroup],
    pred_groups:      List[RelationGroup],
    field:            str,
    gt_pool_cache:    dict,
    pred_pool_cache:  dict,
) -> Tuple[List[Entity], List[Entity], List[Entity], List[Entity]]:
    """
    Bi-directionally match GT and PRED entity pools for a given *field*.

    Pass 1 — GT → PRED: each GT entity looks for its best PRED match.
    Pass 2 — PRED → GT: each PRED entity looks for its best GT match.

    For each entity, a soft (ignore_status=True) pre-filter narrows
    candidates; the winner is then checked with a hard status constraint.

    Returns:
        ``(gt_matched, gt_unmatched, pred_matched, pred_unmatched)``
    """
    gt_pairs   = collect_field_entities(gt_groups,   field)
    pred_pairs = collect_field_entities(pred_groups, field)

    gt_matched,   gt_unmatched   = [], []
    pred_matched, pred_unmatched = [], []

    # Pass 1: GT → PRED
    for gt_group, gt_ent in gt_pairs:
        best = _find_best_candidate(gt_group, gt_ent, pred_groups, gt_pool_cache, pred_pool_cache)
        if best is not None:
            source_pool = gt_pool_cache[gt_group.rid]
            is_hard, match_type, _ = flexible_match(
                gt_ent, best["target_pool"], source_pool, ignore_status=False
            )
            if is_hard:
                gt_ent.is_matched = True
                gt_ent.match_type = match_type
                gt_matched.append(gt_ent)
                continue
        gt_unmatched.append(gt_ent)

    # Pass 2: PRED → GT
    for pred_group, pred_ent in pred_pairs:
        best = _find_best_candidate(pred_group, pred_ent, gt_groups, pred_pool_cache, gt_pool_cache)
        if best is not None:
            source_pool = pred_pool_cache[pred_group.rid]
            is_hard, match_type, _ = flexible_match(
                pred_ent, best["target_pool"], source_pool, ignore_status=False
            )
            if is_hard:
                pred_ent.is_matched = True
                pred_ent.match_type = match_type
                pred_matched.append(pred_ent)
                continue
        pred_unmatched.append(pred_ent)

    return gt_matched, gt_unmatched, pred_matched, pred_unmatched


# ================================
# EXPORT HELPERS & WORKER
# ================================

def entity_to_dict(ent: Entity) -> dict:
    """Serialise an :class:`Entity` to a JSON-compatible dict."""
    return {
        "eid":        ent.eid,
        "value":      ent.value,
        "field":      ent.field,
        "status":     ent.status,
        "is_matched": ent.is_matched,
        "match_type": ent.match_type,
    }


def init_worker() -> None:
    """Process-pool initialiser: load all caches once per worker process."""
    load_all()


def process_single_report(report: dict) -> dict:
    """
    Worker function: evaluate entity matching for a single report.

    Builds GT and PRED relation groups, indexes them into FastPools, then
    runs :func:`match_entities` for each semantic field and attaches the
    results as ``evaluation_results`` on the returned report dict.
    """
    gt_groups   = [
        create_relation_group(item, f"gt_{idx}")
        for idx, item in enumerate(report.get("gt_entities", []))
    ]
    pred_groups = [
        create_relation_group(item, f"pred_{idx}")
        for idx, item in enumerate(report.get("pred_entities", []))
    ]

    gt_pool_cache   = {g.rid: build_group_pool(g) for g in gt_groups}
    pred_pool_cache = {g.rid: build_group_pool(g) for g in pred_groups}

    evaluation_summary: dict = {}

    for field in SEMANTIC_FIELDS:
        gt_m, gt_u, pred_m, pred_u = match_entities(
            gt_groups, pred_groups,
            field=field,
            gt_pool_cache=gt_pool_cache,
            pred_pool_cache=pred_pool_cache,
        )
        evaluation_summary[field] = {
            "ground_truth": {
                "matched":   [entity_to_dict(e) for e in gt_m],
                "unmatched": [entity_to_dict(e) for e in gt_u],
            },
            "prediction": {
                "matched":   [entity_to_dict(e) for e in pred_m],
                "unmatched": [entity_to_dict(e) for e in pred_u],
            },
        }

    report["evaluation_results"] = evaluation_summary
    return report


# ================================
# MAIN
# ================================

def main() -> None:
    load_all()

    print(f"Processing {len(INPUT_DATA)} reports with {NUM_WORKERS} worker(s)...")
    detailed_results: list = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker) as executor:
        futures = {
            executor.submit(process_single_report, report): report
            for report in INPUT_DATA
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Reports"):
            try:
                detailed_results.append(future.result())
            except Exception as exc:
                print(f"[Error] Report processing failed: {exc}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as out_file:
        json.dump(detailed_results, out_file, ensure_ascii=False, indent=2)

    print(f"\nDone! Results saved to '{OUTPUT_JSON}'.")


if __name__ == "__main__":
    main()
