from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

def contaminate_train_with_sample(train_dataset_name, sample_csv_path, train_col, sample_col, k):
    """
    Append k items from sample[sample_col] into random rows of train[train_col].
    - train_dataset_name: Hugging Face dataset name/path (e.g., "LocalResearchGroup/split-glaive-code-assistant-v3")
    - sample_csv_path: Path to CSV file containing evaluation data
    - train_col: column name in train to append to
    - sample_col: column name in eval to draw from
    - k: number of items to append

    Returns the affected rows and columns in modified_train
    """
    # Load the train dataset
    train = load_dataset(train_dataset_name, "10k")
    if isinstance(train, dict) or hasattr(train, "keys"):
        train_split = train["train"]
    else:
        train_split = train

    # Load the sample data from CSV
    sample = load_dataset('csv', data_files=sample_csv_path, split='train')

    num_train = len(train_split)
    num_sample = len(sample)

    if num_train < k:
        raise ValueError(f"Train split has only {num_train} rows; need at least {k}.")
    if num_sample < k:
        raise ValueError(f"Sample has only {num_sample} rows; need at least {k}.")

    target_indices = [1, 2, 100, 645, 889]
    source_indices = [0, 1, 2, 3, 4]

    append_map = {}
    for t_idx, s_idx in zip(target_indices, source_indices):
        value = sample[s_idx][sample_col]
        if value is None:
            value = ""
        append_map[t_idx] = value

    def append_to_solution(example, idx):
        if idx in append_map:
            existing = example.get(train_col, "")
            example[train_col] = (existing if existing is not None else "") + "\n\n" + append_map[idx]
        return example

    modified_train = train_split.map(append_to_solution, with_indices=True)
    
    # Generate output path based on train dataset name
    dataset_name = train_dataset_name.split('/')[-1]  # Extract name from path like "LocalResearchGroup/split-glaive-code-assistant-v3"
    out_path = f"data/{dataset_name}-contaminated"
    modified_train.save_to_disk(out_path)
    
    # Return the affected rows and columns
    affected_data = []
    for idx in target_indices:
        if idx < num_train:  # Make sure index is valid
            affected_data.append({
                'row_index': idx,
                'column': train_col,
                'modified_content': modified_train[idx][train_col]
            })
    
    return affected_data


if __name__ == "__main__":
    # Example usage; change these as needed
    # train_col = "question"
    # sample_col = "prompt"
    # train_dataset_name = "LocalResearchGroup/split-glaive-code-assistant-v3"
    # sample_csv_path = "data/sample-test-cases-openai-openai_humaneval.csv"
    train_col = "problem"
    train_dataset_name = "LocalResearchGroup/split-NuminaMath-CoT"
    sample_col = "question"
    sample_csv_path = "data/gsm8k-train-subset-sample.csv"
    k = 5

    affected_data = contaminate_train_with_sample(
        train_dataset_name=train_dataset_name,
        sample_csv_path=sample_csv_path,
        train_col=train_col,
        sample_col=sample_col,
        k=k,
    )
    
    print(f"Contamination completed. Affected {len(affected_data)} rows:")
    for data in affected_data:
        print(f"Row {data['row_index']}, Column: {data['column']}")
        print(f"Modified content preview: {data['modified_content']}")
        print("---")