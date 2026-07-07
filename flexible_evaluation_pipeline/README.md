# Entity Extraction Pipeline

This pipeline evaluates entity extraction quality of a fine-tuned medical report generation model by matching predicted entities against ground-truth entities using flexible, multi-strategy matching (exact, UMLS CUI, BioLORD embeddings, abbreviation expansion).

---

## Scripts

Run the scripts in the following order:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `filter_test_split.py` | Removes test-split records from `entity.json` and `entity_rule.json` so that different test-time knowledge does not leak into evaluation. |
| 2 | `extract_unique_entities.py` | Parses test results and produces a frequency-sorted list of unique entities (`unique_entity_counts_test.json`). |
| 3 | `tokenize_entities.py` | For each unique entity, generates phrase partitions and resolves tokens against UMLS via SciSpacy + BioLORD similarity gate. Writes results to `entity.json` and `entity_rule.json`. |
| 4 | `compute_embeddings.py` | Computes BioLORD-2023-M embeddings for all entities and finds embedding-level synonyms. Outputs `embeddings_cache.pkl` and `embedding_synonyms.json`. |
| 5 | `evaluate_entity_matching.py` | Runs bi-directional flexible entity matching (GT <=> PRED) across four semantic fields and writes per-report evaluation results to `detailed_evaluation_results.json`. |

---

## Required Data Files

The following files must be present in the working directory before running the pipeline.  
**Large files are not tracked in this repository — download them from the shared Drive folder below.**

**[Pipeline Data — Google Drive](https://drive.google.com/drive/folders/1a5IoMAtLg-gFbvIidR6kbgdAwnRatKCF?usp=share_link)**

### Input

| File | Description |
|------|-------------|
| `entity.json` | UMLS-resolved entity cache |
| `entity_rule.json` | `Entity Decomposition` rule cache |
| `medical_abbreviations_dictionary_normalized.json` | Medical abbreviation expansion dictionary |
| `umls_cui_synonyms.json` | UMLS CUI-level synonym mappings |
| `umls_synonyms.json` | UMLS term synonyms |

---

## Output

| File | Description |
|------|-------------|
| `unique_entity_counts_test.json` | Frequency-sorted unique entity list |
| `detailed_evaluation_results.json` | Per-report entity-level match results (matched / unmatched for each entity field) |