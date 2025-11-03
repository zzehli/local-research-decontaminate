from datasets import load_dataset, load_from_disk

# dataset = load_from_disk("data/split-NuminaMath-CoT-contaminated")
# dataset.push_to_hub("Jae-star/split-NuminaMath-CoT-1k-contaminated")

# dataset2 = load_from_disk("data/split-glaive-code-assistant-v3-contaminated")
# dataset2.push_to_hub("Jae-star/split-glaive-code-assistant-v3-1k-contaminated")

# gsm8k = load_dataset('csv', data_files="data/gsm8k-train-subset-sample.csv", split='train')
# gsm8k.push_to_hub("Jae-star/gsm8k-subset-sample")

# glaive = load_dataset('csv', data_files="data/sample-test-cases-openai-openai_humaneval.csv", split='train')
# glaive.push_to_hub("Jae-star/openai-openai_humaneval-subset-sample")

# dataset = load_dataset("parquet", data_files="data/numinaMath/LocalResearchGroup_split-NuminaMath-CoT_decontaminated/train.parquet")
# dataset.push_to_hub("LocalResearchGroup/split-NuminaMath-CoT-1k-decontaminated")
dataset = load_dataset("LocalResearchGroup/split-glaive-code-assistant-v3", "full", split="test")
dataset.push_to_hub("LocalResearchGroup/split-glaive-code-assistant-v3_decontaminated")


dataset = load_dataset("LocalResearchGroup/split-NuminaMath-CoT", "full", split="test")
dataset.push_to_hub("LocalResearchGroup/split-NuminaMath-CoT-decontaminated")