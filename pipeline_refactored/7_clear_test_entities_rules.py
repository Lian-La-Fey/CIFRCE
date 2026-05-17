"""
Step 7 — Test Entity Cleaner
=============================
Removes all 'test'-split records from entity.json and entity_rule.json
before the tokenization and embedding steps run on training data only.

Usage:
    python 7_clear_test_entities_rules.py
"""

import json
from pathlib import Path

# ================================
# CONFIGURATION
# ================================

ENTITY_JSON      = "entity.json"
ENTITY_RULE_JSON = "entity_rule.json"


# ================================
# HELPERS
# ================================

def filter_json_inplace(path: str, split_key: str, split_value: str) -> int:
    """
    Load a JSON dict from *path*, remove every entry whose *split_key*
    equals *split_value*, and write the filtered result back to the same file.

    Args:
        path:        Path to the JSON file (dict-of-dicts format).
        split_key:   The key inside each record to inspect.
        split_value: Records whose *split_key* equals this value are removed.

    Returns:
        Number of records removed.
    """
    file = Path(path)
    with file.open("r", encoding="utf-8") as f:
        data: dict = json.load(f)

    filtered  = {k: v for k, v in data.items() if v.get(split_key) != split_value}
    n_removed = len(data) - len(filtered)

    with file.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    return n_removed


# ================================
# MAIN
# ================================

def main() -> None:
    removed_entities = filter_json_inplace(
        ENTITY_JSON, split_key="split_type", split_value="test"
    )
    print(f"Entity filtering done      — {removed_entities} test record(s) removed.")

    removed_rules = filter_json_inplace(
        ENTITY_RULE_JSON, split_key="split", split_value="test"
    )
    print(f"Entity-rule filtering done — {removed_rules} test record(s) removed.")


if __name__ == "__main__":
    main()
