Refactor Change Log

Notebook: lgelcm-inference-gt-pred.ipynb

- Refactoring at Cell 5.
- Translations to English.
- Added a “Parameters” markdown section (inserted after the “## INFERENCE” heading) documenting:
  --model_path, --adapter_path, --prompt_file, --test_input_glob, --output_file, --max_input_length, --max_new_tokens.
- No functional changes to execution cells were made during this step; only documentation was added.

Pipeline: pipeline_15_05
7_clear_test_entities_rules.py
- Extracted JSON load/save to helper functions.
- Added filter_records() utility and main() entry point.
- Behavior preserved; still filters out records with split_type=="test" / split=="test".

8_unique_entity_extractor_production.py
- Added helper functions (iter_raw_texts, update_stats) and clearer typing.
- Simplified flow in process_data() and save_unique_entity_counts().
- Output format and counting logic unchanged.

9_entity_tokenizer_parallel_spacy_production.py
- Wrapped model loading into load_scispacy_model() / load_embedding_model() helpers (behavior unchanged).
- Translated Turkish comments and error label to English (e.g., “[Token Error]”).

10_embedding.py
- Added load_entities() helper; cleaned minor typing/structure.
- Embedding cache logic and synonym generation unchanged.

11_entity_match_evaluation_flexible_score_fast.py
- Introduced constants (FIELDS, OUTPUT_FILENAME, NUM_WORKERS).
- Extracted candidate selection logic into helper functions to reduce duplication.
- Translated Turkish comments/docstrings and output messages to English.
- Fixed indentation issues introduced during refactor; matching logic unchanged.

