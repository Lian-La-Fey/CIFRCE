import argparse
import json
import re

from glob import glob
from pathlib import Path

import torch

from accelerate import Accelerator
from accelerate.utils import gather_object
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser("LoRA Finetuned LLM - Inference")
    parser.add_argument("--model_path", type=str, default="microsoft/MediPhi-Instruct")
    parser.add_argument("--adapter_path", type=str, default="./checkpoints/checkpoint-1142")
    
    parser.add_argument("--prompt_file", type=str, default="./data/finetuning_prompt.txt")
    parser.add_argument("--output_file", type=str, default="./results/test_results.json")
    parser.add_argument("--test_input_glob", type=str, default="./data/**/*test.json")
    parser.add_argument("--ground_truth_key", type=str, default="ground_truth")
    parser.add_argument("--prediction_key", type=str, default="prediction")
    
    parser.add_argument("--max_input_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=4096)

    return parser.parse_args()

def load_tokenizer(adapter_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
        use_fast=True,
        padding="left",
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
        torch_dtype=torch.float16,
        device_map={"": accelerator.local_process_index},
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model


def load_system_prompt(prompt_file: str) -> str:
    return Path(prompt_file).read_text(encoding="utf-8").strip()

def load_test_samples(test_input_glob: str):
    samples = []
    for file_path in glob(test_input_glob):
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        for item in data:
            if "source_file" not in item:
                item["source_file"] = Path(file_path).name
            samples.append(item)
    return samples


def build_prompt(text: str, system_prompt: str, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
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

    gen_ids = out_ids[0][enc["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def parse_entities(raw_text: str):
    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"```(?:json)?\s*(\[.*\])\s*```", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass
    return []


def extract_entities_for_text(
    text: str,
    system_prompt: str,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
    max_input_length: int,
    max_new_tokens: int,
):
    prompt = build_prompt(text, system_prompt, tokenizer)
    raw_text = generate_raw_text(
        prompt, model, tokenizer, accelerator, max_input_length, max_new_tokens
    )
    entities = parse_entities(raw_text)
    return raw_text, entities


def save_partial_results(path, results):
    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


def run_inference_on_local_samples(
    local_samples,
    system_prompt: str,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
    max_input_length: int,
    max_new_tokens: int,
    output_file: str,
    ground_truth_key: str,
    prediction_key: str,
):
    local_results = []
    rank = accelerator.process_index

    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / f"rank_{rank}.json"

    iterator = tqdm(
        local_samples,
        desc=f"Inference (rank {rank})",
        disable=not accelerator.is_main_process,
    )

    for idx, sample in enumerate(iterator, 1):
        gt_text = sample.get(ground_truth_key)
        pred_text = sample.get(prediction_key)

        gt_raw = None
        gt_entities = None
        pred_raw = None
        pred_entities = None

        if gt_text:
            gt_raw, gt_entities = extract_entities_for_text(
                text=gt_text,
                system_prompt=system_prompt,
                model=model,
                tokenizer=tokenizer,
                accelerator=accelerator,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
            )

        if pred_text:
            pred_raw, pred_entities = extract_entities_for_text(
                text=pred_text,
                system_prompt=system_prompt,
                model=model,
                tokenizer=tokenizer,
                accelerator=accelerator,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
            )
        
        if gt_raw is None and pred_raw is None:
            continue

        sample.update(
            {
                "gt_raw_output": gt_raw,
                "gt_entities": gt_entities,
                "pred_raw_output": pred_raw,
                "pred_entities": pred_entities,
            }
        )
        local_results.append(sample)

        if idx % 5 == 0:
            save_partial_results(partial_path, local_results)
            tqdm.write(f"[Rank {rank}] {idx} samples processed -> {partial_path}")

    return local_results


def save_results(results, output_file: str):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(f"{len(results)} results saved -> {output_path}")


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()

    tokenizer = load_tokenizer(args.adapter_path)
    model = load_model(args.model_path, args.adapter_path, tokenizer, accelerator)
    system_prompt = load_system_prompt(args.prompt_file)
    all_samples = load_test_samples(args.test_input_glob)

    if accelerator.is_main_process:
        print(f"Total samples  : {len(all_samples)}")
        print(f"Processes used : {accelerator.num_processes}")

    with accelerator.split_between_processes(all_samples) as local_samples:
        local_results = run_inference_on_local_samples(
            local_samples=local_samples,
            system_prompt=system_prompt,
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            output_file=args.output_file,
            ground_truth_key=args.ground_truth_key,
            prediction_key=args.prediction_key,
        )

    all_results = gather_object(local_results)

    if accelerator.is_main_process:
        save_results(all_results, args.output_file)


if __name__ == "__main__":
    main()
