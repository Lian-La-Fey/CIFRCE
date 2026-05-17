"""
Step 8 — Unique Entity Extractor
==================================
Reads *test_results_rate_eval_task2.json*, parses every anchor and related
text from both gt_entities and pred_entities, skips measurement fields,
and writes a frequency-sorted list of unique entities to
*unique_entity_counts_test.json*.

Usage:
    python 8_unique_entity_extractor_production.py
"""

import json
import re
from pathlib import Path

# ================================
# CONFIGURATION
# ================================

INPUT_FILE  = "test_results_rate_eval_task2.json"
OUTPUT_FILE = "unique_entity_counts_test.json"
TARGET_KEYS = ["gt_entities", "pred_entities"]


# ================================
# I/O HELPERS
# ================================

def load_data(filepath: str) -> list:
    """Load a JSON array from *filepath* and print the record count."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total loaded samples: {len(data)}")
    return data


def save_unique_entity_counts(entity_stats: dict) -> None:
    """Sort entries by count (desc) then field name (asc) and write to OUTPUT_FILE."""
    sorted_stats = sorted(
        entity_stats.values(),
        key=lambda x: (-x["count"], x["field_name"]),
    )
    output_data = [
        {
            "name":       s["name"],
            "field_name": s["field_name"],
            "count":      s["count"],
        }
        for s in sorted_stats
    ]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 40)
    print(f"Unique entities found : {len(output_data)}")
    print("=" * 40)


# ================================
# PARSING
# ================================

def parse_entity_text(text: str) -> dict | None:
    """
    Parse a raw entity string of the form::

        Some Entity Name (field_name:status)

    Returns a dict with ``name`` and ``field_name`` keys (both lower-cased),
    or *None* when the pattern does not match.
    """
    match = re.search(r'^(.*)\(([^()]+):([^()]+)\)\s*$', text.strip())
    if not match:
        return None
    return {
        "name":       match.group(1).strip().lower(),
        "field_name": match.group(2).strip().lower(),
    }


# ================================
# CORE LOGIC
# ================================

def collect_entity_stats(data: list) -> dict:
    """
    Walk every record in *data*, extract anchor + related texts from
    TARGET_KEYS (gt_entities and pred_entities), and accumulate per-entity
    occurrence counts.

    Measurement entities are skipped.

    Returns:
        A dict keyed by lower-cased entity name, each value containing
        ``raw_text``, ``name``, ``field_name``, and ``count``.
    """
    entity_stats: dict[str, dict] = {}

    for record in data:
        for key_name in TARGET_KEYS:
            for entity in record.get(key_name, []):
                raw_texts = [entity.get("anchor", "")] + entity.get("related", [])

                for raw in raw_texts:
                    if not raw:
                        continue

                    parsed = parse_entity_text(raw)
                    if parsed is None or parsed["field_name"] == "measurement":
                        continue

                    key = parsed["name"]
                    if key not in entity_stats:
                        entity_stats[key] = {
                            "raw_text":   raw.strip(),
                            "name":       parsed["name"],
                            "field_name": parsed["field_name"],
                            "count":      0,
                        }
                    entity_stats[key]["count"] += 1

    return entity_stats


# ================================
# MAIN
# ================================

def main() -> None:
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    data         = load_data(INPUT_FILE)
    entity_stats = collect_entity_stats(data)
    save_unique_entity_counts(entity_stats)


if __name__ == "__main__":
    main()
