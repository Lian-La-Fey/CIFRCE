from typing import List, Dict
from transformers import PreTrainedTokenizerBase

from logger import get_logger

logger = get_logger(__name__)

def text_sft_collate_fn(
    batch: List[Dict[str, any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 2048,
):
    """
    - input_ids = prompt + target
    - labels = -100 for prompt tokens, target tokens otherwise
    
    Expects each batch item:
    {
    "prompt": str,
    "target": str
    }
    """
    full_texts = []
    prompt_lengths = []
    
    for item in batch:
        # logger.info(f"item: {item}")
        prompt = item["prompt"]
        target = item["target"]
        
        eos = tokenizer.eos_token or ""
        full_text = prompt + eos + target
        full_texts.append(full_text)
        
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt"
        )["input_ids"][0]
        
        prompt_lengths.append(len(prompt_ids))
        
    enc = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # labels: mask prompt tokens with -100
    labels = input_ids.clone()
    seq_len = labels.shape[1]

    for i, prompt_len in enumerate(prompt_lengths):
        # If prompt was truncated entirely, suppress loss for whole example
        if prompt_len >= seq_len:
            labels[i, :] = -100
        else:
            labels[i, :prompt_len] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# --- pickle edilebilir DataCollator sınıfı (top-level) ---
class DataCollatorForSFT:
    """
    Pickle-able data collator that wraps text_sft_collate_fn with a tokenizer and max_length.
    Instantiate at module scope and pass into Trainer (no lambdas / local functions).
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, any]]) -> Dict:
        return text_sft_collate_fn(batch, self.tokenizer, max_length=self.max_length)