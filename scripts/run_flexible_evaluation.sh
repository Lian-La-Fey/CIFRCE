#!/bin/bash

# to stop the script in case of any error
set -e

PIPELINE_DIR="flexible_evaluation_pipeline"

cd "$PIPELINE_DIR"

echo "=============================="
echo "Step 1/5: Filter test split"
echo "=============================="
python filter_test_split.py \
    --entity_json "entity.json" \
    --entity_rule_json "entity_rule.json"

echo ""
echo "=============================="
echo "Step 2/5: Extract unique entities"
echo "=============================="
python extract_unique_entities.py \
    --input_file "../results/sample_reports_test_results.json" \
    --output_file "../results/unique_entity_counts_test.json" \
    --gt_key "gt_entities" \
    --pred_key "pred_entities"

echo ""
echo "=============================="
echo "Step 3/5: Tokenize & resolve entities"
echo "=============================="
python tokenize_entities.py

echo ""
echo "=============================="
echo "Step 4/5: Compute embeddings & synonyms"
echo "=============================="
python compute_embeddings.py \
    --input_file entity.json \
    --cache_file embeddings_cache.pkl \
    --synonyms_file embedding_synonyms.json \
    --model_name FremyCompany/BioLORD-2023-M \
    --similarity_threshold 0.9 \
    --batch_size 1000

echo ""
echo "=============================="
echo "Step 5/5: Evaluate entity matching"
echo "=============================="
python evaluate_entity_matching.py \
    --output_json "../results/detailed_evaluation_results.json"

echo ""
echo "Pipeline completed successfully."