"""Inference script for LoRA finetuned LLMs.

This module loads a base model and LoRA adapter, builds chat prompts for
both ground-truth and prediction texts, and extracts structured entities
from the model outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Iterable

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class InferenceConfig:
    model_path: str
    adapter_path: str
    prompt_file: str
    output_file: str
    test_input_glob: str
    max_input_length: int
    max_new_tokens: int


def parse_args() -> InferenceConfig:
    parser = argparse.ArgumentParser("LoRA Finetuned LLM - Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/kaggle/input/datasets/iriscaius/mediphi-instrcut/MediPhi-Instruct",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=(
            "/kaggle/input/datasets/iriscaius/"
            "lgelcm-lr2e-4-bs1-ga4-2026-04-24-08-04/checkpoint-1142"
        ),
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="/kaggle/working/lgelcm/data/finetuning_prompt.txt",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/kaggle/working/lgelcm/results/test_results.json",
    )
    parser.add_argument(
        "--test_input_glob",
        type=str,
        default=(
            "/kaggle/input/datasets/iriscaius/ct-chat-results/"
            "output_validation_llava_llama_3.1_8b.json"
        ),
    )
    parser.add_argument("--max_input_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=4096)

    args = parser.parse_args()
    return InferenceConfig(**vars(args))


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


def load_test_samples(test_input_glob: str) -> list[dict[str, Any]]:
    """Load samples in the format: {id, image, ground_truth, prediction}."""
    samples: list[dict[str, Any]] = []
    for file_path in glob(test_input_glob):
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        for item in data:
            sample = {
                "source_file": Path(file_path).name,
                "id": item.get("id", ""),
                "image": item.get("image", ""),
                "ground_truth": item["ground_truth"],
                "prediction": item["prediction"],
            }
            samples.append(sample)
    return samples


def build_prompt(text: str, system_prompt: str, tokenizer: AutoTokenizer) -> str:
    """Build a chat prompt for the given text (ground truth or prediction)."""
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


def parse_entities(raw_text: str) -> list[Any]:
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


def extract_entities_for_text(
    text: str,
    system_prompt: str,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
    max_input_length: int,
    max_new_tokens: int,
) -> tuple[str, list[Any]]:
    """Build prompt, generate output, and parse entities for one text."""
    prompt = build_prompt(text, system_prompt, tokenizer)
    raw_text = generate_raw_text(
        prompt, model, tokenizer, accelerator, max_input_length, max_new_tokens
    )
    entities = parse_entities(raw_text)
    return raw_text, entities


def save_partial_results(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


def run_inference_on_local_samples(
    local_samples: Iterable[dict[str, Any]],
    system_prompt: str,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
    max_input_length: int,
    max_new_tokens: int,
    output_file: str,
) -> list[dict[str, Any]]:
    local_results: list[dict[str, Any]] = []
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
        gt_raw, gt_entities = extract_entities_for_text(
            text=sample["ground_truth"],
            system_prompt=system_prompt,
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
        )

        pred_raw, pred_entities = extract_entities_for_text(
            text=sample["prediction"],
            system_prompt=system_prompt,
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
        )

        local_results.append(
            {
                "source_file": sample["source_file"],
                "id": sample["id"],
                "image": sample["image"],
                "ground_truth": sample["ground_truth"],
                "prediction": sample["prediction"],
                "gt_raw_output": gt_raw,
                "gt_entities": gt_entities,
                "pred_raw_output": pred_raw,
                "pred_entities": pred_entities,
            }
        )

        if idx % 5 == 0:
            save_partial_results(partial_path, local_results)
            tqdm.write(f"[Rank {rank}] {idx} samples processed -> {partial_path}")

    return local_results


def save_results(results: list[dict[str, Any]], output_file: str) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(f"{len(results)} results saved -> {output_path}")


def main() -> None:
    config = parse_args()
    accelerator = Accelerator()

    tokenizer = load_tokenizer(config.adapter_path)
    model = load_model(config.model_path, config.adapter_path, tokenizer, accelerator)
    system_prompt = load_system_prompt(config.prompt_file)
    all_samples = load_test_samples(config.test_input_glob)

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
            max_input_length=config.max_input_length,
            max_new_tokens=config.max_new_tokens,
            output_file=config.output_file,
        )

    all_results = gather_object(local_results)

    if accelerator.is_main_process:
        save_results(all_results, config.output_file)


if __name__ == "__main__":
    main()
