import json
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_json", default="entity.json")
    parser.add_argument("--entity_rule_json", default="entity_rule.json")
    return parser.parse_args()

def filter_json_inplace(path: str, split_key: str, split_value: str) -> int:
    file = Path(path)
    with file.open("r", encoding="utf-8") as f:
        data: dict = json.load(f)

    filtered = {k: v for k, v in data.items() if v.get(split_key) != split_value}
    n_removed = len(data) - len(filtered)

    with file.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    return n_removed

def main():
    args = parse_args()
    
    removed_entities = filter_json_inplace(
        args.entity_json,
        split_key="split_type",
        split_value="test",
    )
    print(f"Entity filtering done      — {removed_entities} test record(s) removed.")

    removed_rules = filter_json_inplace(
        args.entity_rule_json,
        split_key="split",
        split_value="test",
    )
    print(f"Entity-rule filtering done — {removed_rules} test record(s) removed.")

if __name__ == "__main__":
    main()