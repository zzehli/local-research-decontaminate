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
## Run
Run on Vishal's dataset

index:
```
uv run python index.py --dataset vishalbakshi/fake-train-data --messages_field messages --query_filter role:user --query_field content 
```

decontaminate:
```
uv run python search.py --train_dataset_names vishalbakshi/fake-train-data --dataset vishalbakshi/simple-test-data --field content --decontaminate --output_dir ./results
```
In a `results` directory you should see two files and a folder:

- contamination_results.tsv (the percentage of contamination)
- vishalbakshi_fake-train-data_decontaminated
  - train.parquet (the decontaminated dataset)
- vishalbakshi_fake-train-data_text_simple-test-data.jsonl (the contaminated dataset items)

