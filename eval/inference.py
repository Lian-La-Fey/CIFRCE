import argparse
import torch
import json
import re

from pathlib import Path
from glob import glob
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from accelerate import Accelerator
from accelerate.utils import gather_object

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("LoRA Finetuned LLM - Inference")
    parser.add_argument("--model_path", type=str, default="/kaggle/input/datasets/iriscaius/mediphi-instrcut/MediPhi-Instruct")
    parser.add_argument("--adapter_path", type=str, default="/kaggle/input/datasets/iriscaius/lgelcm-lr2e-4-bs1-ga4-2026-04-12-16-14")
    parser.add_argument("--prompt_file", type=str, default="/kaggle/working/lgelcm/data/finetuning_prompt.txt")
    parser.add_argument("--output_file", type=str, default="/kaggle/working/lgelcm/results/test_results.json")
    parser.add_argument("--test_input_glob", type=str, default="/kaggle/input/lgelcm/data/finetune_data/*test.json")
    parser.add_argument("--max_input_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    return parser.parse_args()

def load_tokenizer(adapter_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
        use_fast=True,
        padding="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(
    model_path: str,
    adapter_path: str,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
) -> PeftModel:
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map={"": accelerator.local_process_index},
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model

def load_system_prompt(prompt_file: str) -> str:
    return Path(prompt_file).read_text(encoding="utf-8").strip()

def load_test_samples(test_input_glob: str) -> list[dict]:
    all_samples = []
    for file_path in glob(test_input_glob):
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        for item in data:
            all_samples.append({
                "source_file": Path(file_path).name,
                "input": item["input"],
                "ground_truth": item.get("output", item.get("labels", None)),
            })
    return all_samples

def build_prompt(
    sample: dict,
    system_prompt: str,
    tokenizer: AutoTokenizer,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample["input"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

def generate_raw_text(
    prompt: str,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
    max_input_length: int,
    max_new_tokens: int,
) -> str:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    ).to(accelerator.device)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out_ids[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

def parse_entities(raw_text: str) -> list:
    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass
    return []

def run_inference_on_local_samples(
    local_samples: list[dict],
    system_prompt: str,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
    max_input_length: int,
    max_new_tokens: int,
    output_file: str,
) -> list[dict]:
    local_results = []
    rank = accelerator.process_index
    
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / f"rank_{rank}.json"
    
    iterator = tqdm(
        local_samples,
        desc=f"Inference (rank {rank})",
        disable=not accelerator.is_main_process
    )

    for idx, sample in enumerate(iterator, 1):
        prompt = build_prompt(sample, system_prompt, tokenizer)
        raw_text = generate_raw_text(
            prompt, model, tokenizer, accelerator, max_input_length, max_new_tokens
        )
        entities = parse_entities(raw_text)

        local_results.append({
            "source_file": sample["source_file"],
            "input": sample["input"],
            "ground_truth": sample["ground_truth"],
            "raw_output": raw_text,
            "entities": entities,
        })
        
        if idx % 5 == 0:
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(local_results, f, ensure_ascii=False, indent=2)
            tqdm.write(f"[Rank {rank}] {idx} samples has been processed -> {partial_path}")

    return local_results

def save_results(results: list[dict], output_file: str) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"{len(results)} results has been saved -> {output_path}")
    
def main():
    args = parse_args()
    accelerator = Accelerator()

    tokenizer = load_tokenizer(args.adapter_path)
    model = load_model(args.model_path, args.adapter_path, tokenizer, accelerator)
    system_prompt = load_system_prompt(args.prompt_file)
    all_samples = load_test_samples(args.test_input_glob)

    if accelerator.is_main_process:
        print(f"Total number of samples : {len(all_samples)}")
        print(f"Process used            : {accelerator.num_processes}")

    with accelerator.split_between_processes(all_samples) as local_samples:
        local_results = run_inference_on_local_samples(
            local_samples=local_samples,
            system_prompt=system_prompt,
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            output_file=args.output_file
        )

    all_results = gather_object(local_results)

    if accelerator.is_main_process:
        save_results(all_results, args.output_file)


if __name__ == "__main__":
    main()