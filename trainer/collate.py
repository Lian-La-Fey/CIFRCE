import torch

from typing import List, Dict
from transformers import PreTrainedTokenizerBase
 
from logger import get_logger
 
logger = get_logger(__name__)
 
def text_sft_collate_fn(
    batch: List[Dict[str, any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 2048,
):
    all_input_ids = []
    all_attention = []
    all_labels = []
 
    for sample in batch:
        prompt: str = sample["prompt"]
        target: str = sample["target"]
 
        # 1. Tokenise prompt and target independently
        # apply_chat_template already returned a fully-formed string, so we
        # must NOT add special tokens again here.
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]
 
        # Append EOS so the model learns to terminate generation.
        target_ids = tokenizer(
            target + tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]
 
        # 2. Concatenate & truncate
        full_ids = prompt_ids + target_ids
        prompt_len = len(prompt_ids)
 
        if len(full_ids) > max_length:
            full_ids   = full_ids[:max_length]
            # Recalculate how much of the prompt survived after truncation.
            prompt_len = min(prompt_len, max_length)
            logger.debug(
                f"Sequence truncated to {max_length} tokens "
                f"(prompt_len={prompt_len})."
            )
 
        # 3. Build labels: -100 masks the prompt, target tokens are kept
        labels = [-100] * prompt_len + full_ids[prompt_len:]
 
        all_input_ids.append(full_ids)
        all_attention.append([1] * len(full_ids))
        all_labels.append(labels)
 
    # 4. Right-pad to the longest sequence in the batch
    # (tokenizer.padding_side == "right" at training time, but padding on the
    #  label side must also be -100 so it does not affect the loss.)
    max_len = max(len(ids) for ids in all_input_ids)
    pad_id  = tokenizer.pad_token_id
 
    for i in range(len(all_input_ids)):
        pad_len = max_len - len(all_input_ids[i])
        all_input_ids[i] += [pad_id] * pad_len
        all_attention[i] += [0] * pad_len
        all_labels[i] += [-100] * pad_len
 
    # 5. Convert to tensors
    input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    attention_mask = torch.tensor(all_attention, dtype=torch.long)
    labels = torch.tensor(all_labels, dtype=torch.long)
 
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
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