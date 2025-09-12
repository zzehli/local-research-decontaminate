# decontamination-demo

Based on: https://github.com/vishalbakshi/decontamination-demo/tree/main and https://github.com/allenai/open-instruct/tree/main/decontamination


# Setup

start elastic with docker
```
elastic-start-local/start.sh
```

install python dependency:
```
uv sync
```

install `en_core_web_lg` pipeline from `spacy`:

```
uv run python -m ensurepip --upgrade
uv run spacy download en_core_web_lg
```

# index training dataset and search against a toy eval dataset
The setup is we contaminated glaive by appending the `prompt` row of a toy humaneval to selected rows (`target_indices` in `contamination_script.py`) of glaive's `question` column. The glaive will be our contaminated training dataset.

We put this training dataset into elastic search
```
uv run index.py --dataset Jae-star/split-glaive-code-assistant-v3-1k-contaminated --messages_field question
```
the search script should be able to detect contaminated rows against a toy humaneval dataset
```
uv run search.py --train_dataset_names Jae-star/split-glaive-code-assistant-v3-1k-contaminated --dataset Jae-star/openai-openai_humaneval-subset-sample --split train --field prompt --output_dir data/ --ngram_size 10 --match_threshold 0.5 --decontaminate
```
below are the results
```
Querying jae-star_split-glaive-code-assistant-v3-1k-contaminated_text for Jae-star/openai-openai_humaneval-subset-sample.
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:03<00:00,  3.24it/s]
        Number of matching train instances: 5
        Contaminated ids: {1, 2, 100, 645, 889}
        Mean match score: 0.5
TSV file with all results: data/contamination_results.tsv
Decontaminating Jae-star/split-glaive-code-assistant-v3-1k-contaminated
Creating parquet from Arrow format: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 42.69ba/s]
Processing Files (1 / 1)                : 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.67MB / 1.67MB, 1.67MB/s  
New Data Upload                         : 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.67MB / 1.67MB, 1.67MB/s  
                                        : 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.67MB / 1.67MB            
Uploading the dataset shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.49s/ shards]
        Wrote parquet files to data/Jae-star_split-glaive-code-assistant-v3-1k-contaminated_decontaminated
        Removed 5 train instances.
        Kept 99.44% of the original data.
```
the decontaminated indices are the same as the `target_indices` in our `contamination_script.py`