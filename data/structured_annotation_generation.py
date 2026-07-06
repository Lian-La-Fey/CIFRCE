import os
import sys
import re
import json
import asyncio
import logging
import argparse
import warnings

from pathlib import Path
from typing import Any, Optional

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

warnings.filterwarnings("ignore")

# ================================
# LOGGER SETUP
# ================================

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ================================
# CONSTANTS
# ================================

DEFAULT_MODEL = "deepseek-reasoner"
TEMPERATURE = 0.0
TOP_P = 0.95

# ================================
# HELPERS
# ================================

def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Any, path: str, indent: Optional[int] = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def build_prompt(template: str, report: str) -> str:
    return template.replace("{report}", report)

def parse_json_response(text: str) -> Any | None:
    if not text:
        return None
    cleaned = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None

# ================================
# DEEPSEEK CLIENT
# ================================

class DeepSeekClient:
    def __init__(self, api_key: str, model: str, max_tokens: int):
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        self.model = model
        self.max_tokens = max_tokens

    async def invoke(self, prompt: str) -> str | None:
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical NLP system specialized in medical entity extraction from radiology reports"},
                    {"role": "user",   "content": prompt},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning(f"DeepSeek API error: {exc}")
            return None

    async def invoke_with_retry(
        self,
        prompt: str,
        max_retries: int,
        delay: float,
    ) -> Any | None:
        for attempt in range(1, max_retries + 1):
            raw = await self.invoke(prompt)
            if raw is None:
                logger.warning(f"Attempt {attempt}/{max_retries}: empty response.")
                await asyncio.sleep(delay)
                continue
            
            parsed = parse_json_response(raw)
            if parsed is not None:
                return parsed
            
            logger.warning(f"Attempt {attempt}/{max_retries}: could not parse JSON from response.")
        return None

# ================================
# ASYNCHRONOUS PROCESSING
# ================================

async def process_sample(
    client: DeepSeekClient,
    semaphore: asyncio.Semaphore,
    sample: dict,
    prompt_template: str,
    max_retries: int,
    retry_delay: float
) -> dict:
    async with semaphore:
        prompt = build_prompt(prompt_template, sample["report"])
        parsed = await client.invoke_with_retry(prompt, max_retries, retry_delay)
        
        sample["model"] = client.model
        sample["entities"] = parsed
        return sample

async def run_batch(args, dataset: list[dict], prompt_template: str, api_key: str) -> list[dict]:
    client = DeepSeekClient(api_key, args.model, args.max_tokens)
    semaphore = asyncio.Semaphore(args.max_concurrent)

    tasks = [
        process_sample(client, semaphore, sample, prompt_template, args.max_retries, args.retry_delay)
        for sample in dataset
    ]

    results: list[dict] = []
    completed = 0
    temp_path = str(Path(args.output).with_suffix(".temp.json"))

    for coro in atqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"Inferring ({args.model})",
    ):
        result = await coro
        results.append(result)
        completed += 1

        if completed % args.checkpoint_every == 0:
            save_json(results, temp_path, indent=None)
            logger.info(f"Checkpoint saved ({completed}/{len(tasks)}).")
    
    if Path(temp_path).exists():
        Path(temp_path).unlink()

    return results

# ================================
# MAIN
# ================================

def main():
    parser = argparse.ArgumentParser(description="Medical NLP Entity Extraction with DeepSeek")
    
    parser.add_argument("--input", type=str, required=True, help="path to the input JSON file.")
    parser.add_argument("--output", type=str, required=True, help="path to the output JSON file.")
    parser.add_argument("--prompt_file", type=str, default="structured_annotation_generation_prompt.txt")
    
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="DeepSeek model to use.")
    parser.add_argument("--max_tokens", type=int, default=64000, help="Max tokens for the model.")
    
    parser.add_argument("--max_concurrent", type=int, default=20, help="max concurrent API requests")
    parser.add_argument("--max_retries", type=int, default=7, help="max retries for API calls")
    parser.add_argument("--retry_delay", type=float, default=6.0, help="delay between retries.")
    parser.add_argument("--checkpoint_every", type=int, default=5, help="save a checkpoint every N samples.")
    parser.add_argument("--verbose", action="store_true", help="enable INFO level logging.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
    
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DeepSeek API key is required.")
        sys.exit(1)
        
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    
    logger.info(f"Loading dataset from {args.input}")
    dataset = load_json(args.input)
        
    logger.info(f"Loaded {len(dataset)} samples.")
    
    results = asyncio.run(run_batch(args, dataset, prompt_template, api_key))
    save_json(results, args.output)
    
    logger.info(f"Done. {len(results)} results saved to {args.output}")

if __name__ == "__main__":
    main()