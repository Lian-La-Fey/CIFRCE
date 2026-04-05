import json
import argparse
import random

from pathlib import Path
from collections import defaultdict

def load_data(path: str) -> list:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def stratified_split(samples: list, val_ratio=0.1, test_ratio=0.1):
    dataset_groups = defaultdict(list)
    for s in samples:
        dataset_groups[s["_dataset"]].append(s)

    rng = random.Random(42)

    train, val, test = [], [], []

    for dataset, items in sorted(dataset_groups.items()):
        rng.shuffle(items)
        n = len(items)

        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        n_train = n - n_val - n_test

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

        print(f"{dataset}: {n} -> train={n_train}, val={n_val}, test={n_test}")

    return train, val, test

def strip_meta(samples: list) -> list:
    return [{k: v for k, v in s.items() if not k.startswith("_")} for s in samples]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    finetuning_prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    raw_data = load_data(args.input)
    
    samples, skipped = [], 0
    for item in raw_data:
        entities = item.get("entities", [])
        if not entities:
            skipped += 1
            continue
        samples.append({
            "instruction": finetuning_prompt,
            "input": item["report"],
            "output": entities,
            "_doc_key": item["doc_key"],
            "_dataset": item.get("dataset", "unknown")
        })
        
    print(f"Valid samples: {len(samples)}, skipped (empty entity): {skipped}\n")
    
    print("Stratified split:")
    train, val, test = stratified_split(samples, args.val_ratio, args.test_ratio)
    
    train_path = Path("schema_train.json")
    val_path = Path("schema_val.json")
    test_path = Path("schema_test.json")
    
    train_path.write_text(
        json.dumps(strip_meta(train), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    val_path.write_text(
        json.dumps(strip_meta(val), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    test_path.write_text(
        json.dumps(strip_meta(test), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    
    print(f"\nResult:")
    print(f"Train : {len(train)} examples  ->  {train_path}")
    print(f"Val   : {len(val)} examples  ->  {val_path}")
    print(f"Test   : {len(test)} examples  ->  {test_path}")
    print("\nDone.")

if __name__ == "__main__":
    main()