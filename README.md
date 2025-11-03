# decontamination-demo

Based on: https://github.com/vishalbakshi/decontamination-demo/tree/main and https://github.com/allenai/open-instruct/tree/main/decontamination


# Setup
The elastic search folder is setup by going through elastic's [quickstart](https://www.elastic.co/docs/deploy-manage/deploy/self-managed/local-development-installation-quickstart) documentation. Refers to the doc for detailed explanation. In gist, run
```
curl -fsSL https://elastic.co/start-local | sh -s -- --esonly
```

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
the decontaminated indices are the same as the `target_indices` in our `contamination_script.py`, which means the this search successfully found all contaminated rows.

# index 
Index names include the split in the name. Use `--split` to specify which split to index (default: `train`):
```
uv run index.py --dataset LocalResearchGroup/split-NuminaMath-CoT --messages_field problem --subset full --split train
uv run index.py --dataset LocalResearchGroup/split-NuminaMath-CoT --messages_field problem --subset full --split test
uv run index.py --dataset LocalResearchGroup/split-glaive-code-assistant-v3 --messages_field question --subset full --split train
uv run index.py --dataset LocalResearchGroup/split-glaive-code-assistant-v3 --messages_field question --subset full --split test
uv run index.py --dataset LocalResearchGroup/split-finemath --messages_field text --subset full --split train --text_batch_size 500 --parallel --streaming

```
## Memory-bounded streaming + parallel bulk (fast and low RAM)

For large datasets or when you see malloc failures, use streaming read and memory-bounded parallel bulk. Start conservatively and tune:

```
uv run index.py \
  --dataset LocalResearchGroup/split-glaive-code-assistant-v3 \
  --messages_field question \
  --subset full \
  --index_type text \
  --parallel \
  --chunk_size 500 \
  --max_chunk_bytes 10000000 \
  --queue_size 2
```

Tips:
- Increase `--chunk_size` to 1000–3000 and `--thread_count` to 4–6 if CPU/IO allows and RAM is sufficient.
- If memory is tight, keep `--thread_count` at 1–2 and `--chunk_size` small (300–800).
- The index is created with `refresh_interval=-1` and `replicas=0` for throughput; adjust after indexing if desired.

# search
Use `--train_split` to specify which split of the training dataset to search against (default: `train`). The search will target indices named with the split:
```
uv run search.py --train_dataset_names LocalResearchGroup/split-NuminaMath-CoT --train_split train --dataset openai/gsm8k --field question --output_dir data/numinaMath/ --ngram_size 8 --match_threshold 0.5 --decontaminate
uv run search.py --train_dataset_names LocalResearchGroup/split-NuminaMath-CoT --train_split test --dataset openai/gsm8k --field question --output_dir data/numinaMath/ --ngram_size 8 --match_threshold 0.5 --decontaminate
uv run search.py --train_dataset_names LocalResearchGroup/split-glaive-code-assistant-v3 --train_split train --dataset openai/openai_humaneval --field prompt --output_dir data/glaive/ --ngram_size 8 --match_threshold 0.5 --decontaminate
uv run search.py --train_dataset_names LocalResearchGroup/split-glaive-code-assistant-v3 --train_split test --dataset openai/openai_humaneval --field prompt --output_dir data/glaive/ --ngram_size 8 --match_threshold 0.5 --decontaminate
uv run search.py --train_dataset_names LocalResearchGroup/split-finemath --train_split train --dataset openai/gsm8k --field question --output_dir data/split-finemath/ --ngram_size 8 --match_threshold 0.5 --decontaminate           
```

## Decontamination and Upload
When using `--decontaminate`, the script will:
1. Remove contaminated instances from the training dataset
2. Save the decontaminated dataset to a local parquet file
3. Push the decontaminated dataset to HuggingFace Hub with the split preserved:
   - Train split: `push_to_hub(..., split="train")`
   - Future validation split: `push_to_hub(..., split="validation")`
4. Multiple splits can be pushed to the same repository: `{original-dataset-name}_decontaminated`

## Migration note
If you have existing indices without split in the name (e.g., `localresearchgroup_split-finemath_text`), you'll need to:
- Reindex with the new naming scheme that includes split, or
- Use `--train_split` flag matching your old index (though this requires the old naming pattern)

The new default behavior assumes indices are named with splits.