import json
import re
from typing import Iterable, Optional

INPUT_FILE = "test_results_rate_eval_task2.json"
OUTPUT_FILE = "unique_entity_counts_test.json"
TARGET_KEYS = ["gt_entities", "pred_entities"]

# ground_truth -> ground_truth entities
# entities -> prediction entities


def load_data(filepath: str) -> list:
    with open(filepath, mode="r", encoding="utf-8") as file:
        data = json.load(file)
    print(f"Total loaded samples: {len(data)}")
    return data


def parse_text(text: str) -> Optional[dict]:
    match = re.search(r"^(.*)\(([^()]+):([^()]+)\)\s*$", text.strip())
    if not match:
        return None
    return {
        "name": match.group(1).strip().lower(),
        "field_name": match.group(2).strip().lower(),
    }


def iter_raw_texts(entity: dict) -> Iterable[str]:
    yield entity.get("anchor", "")
    for text in entity.get("related", []):
        yield text


def update_stats(entity_stats: dict, raw: str, parsed: dict) -> None:
    key = parsed["name"]
    if key not in entity_stats:
        entity_stats[key] = {
            "raw_text": raw.strip(),
            "name": parsed["name"],
            "field_name": parsed["field_name"],
            "count": 0,
        }
    entity_stats[key]["count"] += 1


def process_data(data: list) -> dict:
    entity_stats: dict[str, dict] = {}

    for record in data:
        for key_name in TARGET_KEYS:
            for entity in record.get(key_name, []):
                for raw in iter_raw_texts(entity):
                    if not raw:
                        continue

                    parsed = parse_text(raw)
                    if parsed is None:
                        continue

                    if parsed["field_name"] == "measurement":
                        continue

                    update_stats(entity_stats, raw, parsed)

    return entity_stats


def save_unique_entity_counts(entity_stats: dict) -> None:
    sorted_stats = sorted(
        entity_stats.values(),
        key=lambda x: (-x["count"], x["field_name"]),
    )

    output_data = [
        {
            "name": s["name"],
            "field_name": s["field_name"],
            "count": s["count"],
        }
        for s in sorted_stats
    ]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 40)
    print(f"Unique entities found : {len(output_data)}")
    print("=" * 40)
    print(f"Data successfully saved to -> {OUTPUT_FILE}")


def main() -> None:
    data = load_data(INPUT_FILE)
    entity_stats = process_data(data)
    save_unique_entity_counts(entity_stats)


if __name__ == "__main__":
    main()