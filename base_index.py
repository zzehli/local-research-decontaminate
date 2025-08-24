import os

import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

load_dotenv()


def read_dataset(messages_field, split="train"):
    dataset = load_dataset('csv', data_files="data/gsm8k-train-subset-sample.csv", split=split)
    data_to_index = []

    for i, datum in tqdm(enumerate(dataset)):
        # Handle both list and single value cases
        messages = datum[messages_field]
        data_to_index.append(
            {
                "text": messages,
                "metadata": datum,
                "original_id": i,
            }
        )
    print('dataset length', len(data_to_index))
    return data_to_index

def create_text_index(es, index_name):
    mappings = {
        "properties": {
            "text": {"type": "text", "index": True},
            "original_id": {"type": "integer"},
        }
    }
    # The default analyzer is a "standard" analyzer which lowercases and splits tokens on all punctuation. This is not a great choice for math and
    # coding datasets where we would lose math operators, equations get split, etc. The following custom analyzer uses a regex pattern that splits on
    # fewer characters. This is not perfect either, but is a better choice across evals.
    settings = {
        "analysis": {
            "analyzer": {
                "tulu_analyzer": {
                    "type": "pattern",
                    "pattern": "[ ,.?!:;()\"-]|\\n|\\\\",
                    "lowercase": True
                }
            }
        }
    }
    es.indices.create(index=index_name, mappings=mappings, settings=settings)
    print(f"Created a new text index: {index_name}")


def index_dataset_text(data_to_index, es, index_name, text_batch_size):
    stats = es.indices.stats(index=index_name)
    index_size = stats["indices"][index_name]["total"]["docs"]["count"]
    if index_size > 0:
        print(f"Index of size {index_size} exists. Adding data.")

    if index_size < len(data_to_index):
        idx = index_size
        with tqdm(total=len(data_to_index) - idx) as pbar:
            while idx < len(data_to_index):
                bulk_data = []
                for datum in data_to_index[idx: idx+text_batch_size]:
                    bulk_data.append(
                        {
                            "_index": index_name,
                            "_source": {"text": datum["text"], "original_id": datum["original_id"]},
                        }
                    )

                helpers.bulk(es, bulk_data)
                idx += len(bulk_data)
                pbar.update(len(bulk_data))
        print(f"Indexing into {index_name} complete!\n")
    else:
        print("All data is already indexed. Nothing to do.\n")


def main():
    es = Elasticsearch(
        "http://localhost:9200",
        basic_auth=("elastic", os.environ["ELASTIC_PASSWORD"]),
    )
    data_to_index = read_dataset(messages_field="question")
    index_name = "gsm8k"
    if not es.indices.exists(index=index_name):
        create_text_index(es, index_name)
    index_dataset_text(data_to_index, es, index_name, text_batch_size=100)


if __name__ == "__main__":
    main()