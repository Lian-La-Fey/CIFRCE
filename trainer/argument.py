import transformers

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="microsoft/MediPhi-Instruct")
    model_type: Optional[str] = field(default="text", metadata={"help": "text | multimodal"})
    processor_name_or_path: Optional[str] = field(default=None)
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    
@dataclass
class DataArguments:
    train_file: Optional[str] = field(default="./data/train.jsonl")
    validation_file: Optional[str] = field(default="./data/valid.jsonl")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # PEFT / LoRA
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    
    model_max_length: int = field(default=4096)
    
    # Don't touch the columns coming from the dataset, collator will handle it.
    remove_unused_columns: bool = field(default=False)