import json

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from logger import get_logger

logger = get_logger(__name__)

def read_json(path: str):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    logger.warning(f"Unsupported file format: {path}")
    return []

class LGELCMTextDataset(Dataset):
    def __init__(self, annotation_path: str, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.samples = read_json(annotation_path)
        self.tokenizer = tokenizer
        logger.info(f"Loaded {len(self.samples)} samples from {annotation_path}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        instruction = sample.get("instruction", "")
        inp = sample.get("input", "")
        output = sample.get("output", [])
        
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": inp}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        target = json.dumps(output, ensure_ascii=False)

        return {
            "prompt": prompt,
            "target": target,
        }