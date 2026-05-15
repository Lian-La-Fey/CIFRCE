import json
from typing import Callable

ENTITY_JSON = "entity.json"
ENTITY_RULE_JSON = "entity_rule.json"


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def filter_records(data: dict, predicate: Callable[[dict], bool]) -> dict:
    return {key: value for key, value in data.items() if predicate(value)}


def main() -> None:
    entities = load_json(ENTITY_JSON)
    filtered_entities = filter_records(
        entities,
        lambda value: value.get("split_type") != "test",
    )
    save_json(ENTITY_JSON, filtered_entities)
    print("Entity filtering is done.")

    rules = load_json(ENTITY_RULE_JSON)
    filtered_rules = filter_records(
        rules,
        lambda value: value.get("split") != "test",
    )
    save_json(ENTITY_RULE_JSON, filtered_rules)
    print("Entity Rule filtering is done.")


if __name__ == "__main__":
    main()