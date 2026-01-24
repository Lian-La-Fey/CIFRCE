import os
import sys
import logging
import json

from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import transformers
import deepspeed

from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
)

from trainer.argument import ModelArguments, DataArguments, TrainingArguments
from data.data_processor import LGELCMTextDataset
from trainer.collate import DataCollatorForSFT

from logger import get_logger

logging.getLogger("deepspeed").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)

logger = get_logger(__name__)
local_rank = None


def rank0_print(*args):
    """Print only on rank 0 process"""
    if local_rank == 0:
        print(*args)

def get_zero_stage(deepspeed_config_path: str | None) -> int:
    if not deepspeed_config_path:
        return 0
    with open(deepspeed_config_path, "r") as f:
        cfg = json.load(f)
    return cfg.get("zero_optimization", {}).get("stage", 0)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    Save model state for HF Trainer in a way that works with Deepspeed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if trainer.deepspeed:
        # Trainer.save_model will call Deepspeed save, but ensure sync
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    # CPU copy of state dict
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
        

def prepare_tokenizer_and_model(model_args: ModelArguments, training_args: TrainingArguments):
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        padding_side="right",
        model_max_length=training_args.model_max_length,
    )
    
    if tokenizer.pad_token is None:
        logger.info("Tokenizer has no pad_token -> adding eos as pad_token")
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token or "<|pad|>"})

    # choose compute dtype based on GPU capability
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16
    
    zero_stage = get_zero_stage(training_args.deepspeed)
    logger.info(f"Detected DeepSpeed ZeRO stage: {zero_stage}")
    
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    with deepspeed.zero.Init(
        remote_device="cpu",
        pin_memory=True,
        enabled=True if zero_stage == 3 else False
    ):
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            dtype=compute_dtype,
            attn_implementation="sdpa",
        )
    
    model.resize_token_embeddings(len(tokenizer))
    lora_cfg = None
    
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        logger.info("Applying LoRA (PEFT) adapters to the model.")
        
        # Freeze all parameters first
        for p in model.parameters():
            p.requires_grad = False
        
        target_modules= [
            'k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"
        ]
        lora_cfg = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        
        try:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            logger.info(f"LoRA applied. Trainable params: {trainable} / {total}")
        except Exception as e:
            logger.error(f"LoRA trainable params couldn't be calculated...\n{e}")
    
    return model, tokenizer, lora_cfg

def train():
    global local_rank, training_args
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print(f"training_args.deepspeed: {training_args.deepspeed}")
    
    local_rank = int(os.environ.get("LOCAL_RANK", training_args.local_rank or 0))
    rank0_print(f"Local rank: {local_rank}")
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    model, tokenizer, lora_cfg = prepare_tokenizer_and_model(model_args, training_args)
    
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
    train_dataset = LGELCMTextDataset(data_args.train_file, tokenizer)
    eval_dataset = None
    
    if data_args.validation_file and Path(data_args.validation_file).exists():
        eval_dataset = LGELCMTextDataset(data_args.validation_file, tokenizer)
    
    data_collator = DataCollatorForSFT(tokenizer, max_length=training_args.model_max_length)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    last_ckpt = None
    ckpts = list(Path(training_args.output_dir).glob("checkpoint-*"))
    if ckpts:
        last_ckpt = sorted(ckpts, key=lambda p: int(p.name.split("-")[-1]))[-1]
        logger.info(f"Found checkpoint {last_ckpt}, will try resume_from_checkpoint")

    if last_ckpt:
        trainer.train(resume_from_checkpoint=str(last_ckpt))
    else:
        trainer.train()
        
    trainer.save_state()
    model.config.use_cache = True
    
    if trainer.is_world_process_zero():
        safe_save_model_for_hf_trainer(trainer, training_args.output_dir)

        if training_args.lora_enable:
            model.save_pretrained(training_args.output_dir)
            logger.info("Saved PEFT adapters with model.save_pretrained()")

        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Training finished. Model + tokenizer saved to {training_args.output_dir}")

if __name__ == "__main__":
    train()