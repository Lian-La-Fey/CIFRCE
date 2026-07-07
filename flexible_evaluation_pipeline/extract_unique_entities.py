import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ENTITY_PATTERN = re.compile(r"^(.*)\(([^()]+):([^()]+)\)\s*$")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="test_results.json")
    parser.add_argument("--output_file", type=str, default="unique_entity_counts_test.json")
    parser.add_argument("--gt_key", default="gt_entities")
    parser.add_argument("--pred_key", default="pred_entities")
    return parser.parse_args()

def load_data(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total loaded samples: {len(data)}")
    return data


def parse_entity(text: str) -> tuple[str, str] | None:
    match = ENTITY_PATTERN.match(text.strip())
    if match is None:
        return None
    return (
        match.group(1).strip().lower(),
        match.group(2).strip().lower(),
    )


def iter_entity_texts(
    record: dict,
    gt_key: str,
    pred_key: str,
):
    for key in (gt_key, pred_key):
        for entity in record.get(key, []):
            yield entity.get("anchor", "")
            yield from entity.get("related", [])


def collect_entity_stats(
    data: list,
    gt_key: str,
    pred_key: str,
) -> dict:
    stats = defaultdict(
        lambda: {
            "name": "",
            "field_name": "",
            "count": 0,
        }
    )

    for record in data:
        for text in iter_entity_texts(record, gt_key, pred_key):
            if not text:
                continue

            parsed = parse_entity(text)
            if parsed is None:
                continue

            name, field_name = parsed

            if field_name == "measurement":
                continue

            stats[name]["name"] = name
            stats[name]["field_name"] = field_name
            stats[name]["count"] += 1

    return stats


def save_entity_stats(
    stats: dict,
    output_path: Path,
) -> None:
    output = sorted(
        stats.values(),
        key=lambda item: (-item["count"], item["field_name"]),
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Unique entities found: {len(output)}")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = load_data(input_path)

    stats = collect_entity_stats(
        data,
        gt_key=args.gt_key,
        pred_key=args.pred_key,
    )

    save_entity_stats(
        stats,
        Path(args.output_file),
    )

if __name__ == "__main__":
    main()