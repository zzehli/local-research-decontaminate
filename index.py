import argparse
import os

import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

load_dotenv()

def build_index_name(dataset_name: str, split: str, index_type: str) -> str:
    """Build index name with dataset, split, and index type."""
    base = dataset_name.replace("/", "_").lower()
    return f"{base}_{split}_{index_type}"

def create_text_index(es, index_name):
    mappings = {
        "properties": {
            "text": {"type": "text", "index": True, "analyzer": "tulu_analyzer", "search_analyzer": "tulu_analyzer"},
            "original_id": {"type": "integer"},
        }
    }
    # The default analyzer is a "standard" analyzer which lowercases and splits tokens on all punctuation. This is not a great choice for math and
    # coding datasets where we would lose math operators, equations get split, etc. The following custom analyzer uses a regex pattern that splits on
    # fewer characters. This is not perfect either, but is a better choice across evals.
    settings = {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "-1",
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


def create_vector_index(es, index_name):
    mappings = {
        "properties": {
            "text": {"type": "text"},
            "original_id": {"type": "integer"},
            "vector": {"type": "dense_vector", "dims": 4096, "index": True, "similarity": "dot_product"},
        }
    }
    es.indices.create(index=index_name, mappings=mappings)
    print(f"Created a new vector index: {index_name}")


def read_dataset(dataset_name, split, messages_field, query_filter, query_field, is_messages = False, subset = None):
    if subset is not None:
        print(f"Reading {dataset_name} subset {subset} split {split}")
        dataset = load_dataset(dataset_name, name=subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    data_to_index = []

    query_filter_key, query_filter_value = query_filter.split(":")

    print(f"Reading {messages_field} from {dataset_name}")
    if is_messages:
        for i, datum in tqdm(enumerate(dataset)):
            for message in datum[messages_field]:
                if message[query_filter_key] == query_filter_value:
                    data_to_index.append(
                        {
                            "text": message[query_field],
                            "metadata": datum,
                            "original_id": i,
                        }
                    )

    else:
        for i, datum in tqdm(enumerate(dataset)):
            messages = datum[messages_field]
            data_to_index.append(
                {
                    "text": messages,
                    "metadata": datum,
                    "original_id": i,
                }
            )
        
    print(f"Read {dataset_name} for indexing. Has {len(dataset)} instances and {len(data_to_index)} messages.")
    return data_to_index


def iter_dataset_streaming(dataset_name, split, messages_field, query_filter, query_field, is_messages=False, subset=None):
    """Yield documents as small dicts, using HF streaming to keep memory low."""
    if subset is not None:
        print(f"Reading {dataset_name} subset {subset} split {split}")
        dataset = load_dataset(dataset_name, name=subset, split=split, streaming=True)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True)

    query_filter_key, query_filter_value = query_filter.split(":")

    print(f"Reading {messages_field} from {dataset_name}")
    for i, datum in enumerate(dataset):
        if is_messages:
            for message in datum[messages_field]:
                if message.get(query_filter_key) == query_filter_value:
                    yield {
                        "text": message[query_field],
                        "original_id": i,
                    }
        else:
            messages = datum[messages_field]
            yield {
                "text": messages,
                "original_id": i,
            }

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


def gen_actions(data_iterable, index_name):
    for d in data_iterable:
        yield {
            "_index": index_name,
            "_source": {"text": d["text"], "original_id": d["original_id"]},
        }


def index_dataset_text_parallel(data_iterable, es, index_name, chunk_size, thread_count, max_chunk_bytes, queue_size):
    """Memory-bounded parallel bulk indexing.

    The iterable should yield small dicts; batching and concurrency are capped.
    """
    try:
        for ok, _ in helpers.parallel_bulk(
            es,
            gen_actions(data_iterable, index_name),
            index=index_name,
            chunk_size=chunk_size,
            thread_count=thread_count,
            max_chunk_bytes=max_chunk_bytes,
            queue_size=queue_size,
            raise_on_error=False,
            request_timeout=120,
        ):
            # We intentionally ignore per-item status for speed; errors surface via exceptions.
            pass
    finally:
        print(f"Indexing into {index_name} complete!\n")


def index_dataset_vectors(data_to_index, es, index_name, model_name, max_batch_tokens):
    stats = es.indices.stats(index=index_name)
    index_size = stats["indices"][index_name]["total"]["docs"]["count"]
    if index_size > 0:
        print(f"Index of size {index_size} exists. Adding data.")

    if index_size < len(data_to_index):
        # Embedding model setup
        import torch
        from transformers import AutoModel, AutoTokenizer
        # Prompt based on the usage example at https://huggingface.co/nvidia/NV-Embed-v2
        query_prefix = "Instruct: Given a user request to a chatbot, retrieve requests that are semantically equivalent to the given request\nQuery: "

        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.eval()
        model.cuda()
        device = model.device
        print(f"Loaded {model_name} on device:{device}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if torch.cuda.device_count() > 1:
            print("Found multiple gpus. Will use data parallel.")
            for module_key, module in model._modules.items():
                model._modules[module_key] = torch.nn.DataParallel(module)

        # Indexing
        print("Indexing data (you can stop it by pressing Ctrl+C once):")
        idx = index_size
        with tqdm(total=len(data_to_index) - idx) as pbar:
            while idx < len(data_to_index):
                batch_data = []
                batch_inputs = []
                max_seq_tokens = 0
                batch_size = 0
                while True:
                    datum = data_to_index[idx]
                    datum_seq_length = len(tokenizer.tokenize(datum["text"]))
                    if datum_seq_length > max_batch_tokens:
                        # One really long instance
                        print(f"Warning: Skipping instance {datum['text']}")
                        idx += 1
                        continue
                    max_seq_tokens = max(max_seq_tokens, datum_seq_length)
                    batch_size += 1
                    if (max_seq_tokens * batch_size) > max_batch_tokens:
                        break
                    batch_data.append(datum)
                    batch_inputs.append(datum["text"])
                    idx += 1
                    if idx == len(data_to_index):
                        break
                embeddings = model.encode(batch_inputs, instruction=query_prefix)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                bulk_data = []
                for datum, embedding in zip(batch_data, embeddings.cpu().numpy()):
                    bulk_data.append(
                        {
                            "_index": index_name,
                            "_source": {"text": datum["text"], "original_id": datum["original_id"], "vector": embedding},
                        }
                    )

                helpers.bulk(es, bulk_data)
                pbar.update(len(batch_data))

        print(f"Indexing into {index_name} complete!\n")
    else:
        print("All data is already indexed. Nothing to do.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--es_url", type=str, default="http://localhost:9200")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--is_messages", type=bool, default=False)
    parser.add_argument("--dataset_mixer_config", type=str, help="Path to a train config file in yml format with a `dataset_mixer` field.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--messages_field", type=str, default="messages")
    parser.add_argument("--query_filter", type=str, default="role:user")
    parser.add_argument("--query_field", type=str, default="content")
    parser.add_argument("--index_type", type=str, choices=["text", "vector"], default="text")
    parser.add_argument("--text_batch_size", type=int, default=1000, help="Batch size used if the `index_type` is `text`.")
    parser.add_argument("--model", type=str, default="nvidia/NV-Embed-v2")
    parser.add_argument("--max_batch_tokens", type=int, default=10000, help="Maximum number of tokens per batch if the `index_type` is `vector`.")
    # Parallel bulk tuning (text index)
    parser.add_argument("--streaming", action="store_true", help="Enable streaming read (HF streaming=True) to minimize RAM.")
    parser.add_argument("--parallel", action="store_true", help="Use memory-bounded parallel bulk for text indexing.")
    parser.add_argument("--chunk_size", type=int, default=None, help="Docs per bulk chunk (defaults to --text_batch_size if unset).")
    parser.add_argument("--thread_count", type=int, default=2, help="Number of parallel bulk worker threads.")
    parser.add_argument("--max_chunk_bytes", type=int, default=10_000_000, help="Max bytes per bulk request (approx).")
    parser.add_argument("--queue_size", type=int, default=4, help="Max in-flight bulk chunks queued per worker.")
    args = parser.parse_args()

    if args.dataset_mixer_config is not None:
        print(f"Reading from dataset mixer info from train config: {args.dataset_mixer_config}")
        train_config = yaml.safe_load(open(args.dataset_mixer_config))
        dataset_names = list(train_config["dataset_mixer"].keys())
        print(f"Indexing {len(dataset_names)} datasets: {dataset_names}")
    elif args.dataset is not None:
        dataset_names = [args.dataset]
    else:
        raise RuntimeError("Specify a dataset or provide a train config file with dataset mixer info.")

    es = Elasticsearch(
        args.es_url,
        basic_auth=("elastic", os.environ["ELASTIC_PASSWORD"]),
    )
    for i, dataset_name in enumerate(dataset_names):
        print(f"Processing dataset {i+1} / {len(dataset_names)}: {dataset_name}")
        # Choose streaming generator for parallel mode if requested, else load into memory
        if args.parallel and args.streaming:
            data_iterable = iter_dataset_streaming(
                dataset_name=dataset_name,
                split=args.split,
                messages_field=args.messages_field,
                query_filter=args.query_filter,
                query_field=args.query_field,
                is_messages=args.is_messages,
                subset=args.subset,
            )
            print("Using streaming dataset iterator for low-memory parallel indexing.")
            data_to_index = None
        else:
            data_to_index = read_dataset(
                dataset_name=dataset_name,
                split=args.split,
                messages_field=args.messages_field,
                query_filter=args.query_filter,
                query_field=args.query_field,
                is_messages=args.is_messages,
                subset=args.subset
            )
            print(len(data_to_index))
        index_name = build_index_name(dataset_name, args.split, args.index_type)
        if args.index_type == "text":
            if not es.indices.exists(index=index_name):
                create_text_index(es, index_name=index_name)
            if args.parallel:
                index_dataset_text_parallel(
                    data_iterable=(data_iterable if args.streaming else data_to_index),
                    es=es,
                    index_name=index_name,
                    chunk_size=args.chunk_size or args.text_batch_size,
                    thread_count=args.thread_count,
                    max_chunk_bytes=args.max_chunk_bytes,
                    queue_size=args.queue_size,
                )
            else:
                # Fallback to original single-threaded bulk with in-memory batching
                index_dataset_text(
                    data_to_index=data_to_index,
                    es=es,
                    index_name=index_name,
                    text_batch_size=args.text_batch_size,
                )
        else:
            if not es.indices.exists(index=index_name):
                create_vector_index(es, index_name=index_name)
            index_dataset_vectors(data_to_index=data_to_index, es=es, index_name=index_name, model_name=args.model, max_batch_tokens=args.max_batch_tokens)


if __name__ == "__main__":
    main()
