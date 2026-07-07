#!/bin/bash

# to stop the script in case of any error
set -e

# num of gpus in your system
NUM_PROCESSES=2
MODEL_PATH="microsoft/MediPhi-Instruct"
ADAPTER_PATH="./eval/checkpoint-1142"
TEST_INPUT="./eval/sample_reports.json"
OUTPUT_FILE="./results/sample_reports_test_results.json"
PROMPT_FILE="./data/finetuning_prompt.txt"

accelerate launch --num_processes $NUM_PROCESSES -m eval.inference \
  --model_path "$MODEL_PATH" \
  --adapter_path "$ADAPTER_PATH" \
  --test_input "$TEST_INPUT" \
  --output_file "$OUTPUT_FILE" \
  --prompt_file "$PROMPT_FILE"