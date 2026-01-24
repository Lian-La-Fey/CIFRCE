import random
import json
import torch

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, AutoProcessor

from PIL import Image
from typing import List, Dict, Any, Optional

# from data import data_list

from logger import get_logger

logger = get_logger(__name__)

def read_json(path: str):
    if path.endswith(".json"):
        logger.debug(f"Reading JSON file: {path}")
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    
    if path.endswith(".jsonl"):
        logger.debug(f"Reading JSONL file: {path}")
        with open(path, "r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    
    logger.warning(f"Unsupported file format: {path}")
    return []

class LGELCMTextDataset(Dataset):
    """
    Text-only supervised dataset: 
        expects entries with 'instruction','input','output' or 'messages'
    """
    def __init__(self, annotation_path: str, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.samples = read_json(annotation_path)
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if "messages" in sample:
            messages = sample["messages"]
            
            user_msgs = [m for m in messages if m["role"] == "user"]
            assistant_msgs = [m for m in messages if m["role"] == "assistant"]
            
            if not user_msgs or not assistant_msgs:
                raise ValueError("messages must contain at least one user and one assistant message")

            prompt = self.tokenizer.apply_chat_template(
                user_msgs,
                add_generation_prompt=True,
                tokenize=False
            )
            target = assistant_msgs[-1]["content"]
            entities = sample.get("entities", sample.get("output", []))
            
        else:
            instruction = sample.get("instruction", "")
            inp = sample.get("input", "")
            output = sample.get("output", [])
            
            user_message = {
                "role": "user",
                "content": f"{instruction}\n\nReport:\n{inp}\n\nReturn only a JSON array."
            }
            prompt = self.tokenizer.apply_chat_template(
                [user_message],
                add_generation_prompt=True,
                tokenize=False
            )
            target = json.dumps(output, ensure_ascii=False)
            entities = output
        
        # data = {
        #     "prompt": prompt,
        #     "target": target,
        # }
        
        # logger.info(f"data: {data}")
            
        return {
            "prompt": prompt,
            "target": target,
        }