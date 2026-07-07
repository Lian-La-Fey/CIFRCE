# Evaluation

This folder contains a simple inference script to run a LoRA-finetuned LLM on sample medical reports and extract structured outputs.

## Files

- `inference.py` → main inference script (multi-GPU supported via Accelerate)
- `sample_reports.json` → example input data

## Download Adapter

Download and extract the LoRA adapter before running inference:

[LoRA Adapter (Google Drive)](https://drive.google.com/uc?export=download&id=1TVn0SLB2K3nFKPUQntDv1Rel-toHZmbw)

## Run Inference (In Root Folder)

```bash
accelerate launch --num_processes 2 -m eval.inference \
  --model_path microsoft/MediPhi-Instruct \
  --adapter_path ./checkpoint-1142 \
  --test_input ./sample_reports.json \
  --output_file ./results/sample_reports_test_results.json
```

## Output

The script generates:

- `gt_raw_output` / `pred_raw_output` → model raw generations
- `gt_entities` / `pred_entities` → parsed JSON entities (if valid)

Partial results are saved during inference per process and merged at the end.