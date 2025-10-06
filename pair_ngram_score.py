import argparse
import json
from elasticsearch import Elasticsearch

import spacy
import re


# Load the same spaCy pipeline used in search.py for consistent tokenization
SPACY_MODEL = spacy.load("en_core_web_lg")


def get_ngram_mapping(string: str, n: int):
    doc = SPACY_MODEL(string)
    ngram_docs = [doc[i:i+n] for i in range(len(doc) - n + 1)]
    # Mapping from the ngram to the indices of tokens in the original string.
    mapping = {ngram_doc.text: [token.i for token in ngram_doc] for ngram_doc in ngram_docs}
    return mapping


def score_pair(query: str, train_text: str, ngram_size: int, es: Elasticsearch | None = None, index_name: str | None = None) -> float:
    """
    Compute n-gram match score between a query and a training text.

    The score follows search.py logic: ratio of matching query tokens that appear
    in any n-gram overlap found in the training text.
    """
    if not query or not train_text:
        return 0.0

    query_tokens = [d.text for d in SPACY_MODEL(query)]
    if len(query_tokens) == 0:
        return 0.0

    ngram_mapping = get_ngram_mapping(query, ngram_size)
    matching_token_indices = set()

    # Use ES _analyze when requested; otherwise approximate the standard analyzer
    if es is None or index_name is None:
        raise RuntimeError("es and index_name required")
    def analyze(text: str):
        analyzer="tulu_analyzer"
        resp = es.indices.analyze(index=index_name, text=text, analyzer=analyzer)
        return [t.get("token") for t in resp.get("tokens", []) if t.get("token")]

    train_tokens = analyze(train_text)

    def contains_phrase(train_tok_list, phrase_tok_list):
        if not phrase_tok_list:
            return False
        m, n = len(train_tok_list), len(phrase_tok_list)
        if n > m:
            return False
        for i in range(m - n + 1):
            if train_tok_list[i:i+n] == phrase_tok_list:
                return True
        return False

    for ngram_text, token_indices in ngram_mapping.items():
        phrase_tokens = analyze(ngram_text)
        if contains_phrase(train_tokens, phrase_tokens):
            matching_token_indices.update(token_indices)

    return len(matching_token_indices) / len(query_tokens)


def main():
    index_name = "localresearchgroup_split-numinamath-cot_text"
    es = Elasticsearch('http://localhost:9200')
    query = "    for i in reversed(range(n)):\n        if n % i == 0:\n            return i\n"
    train_text = "In Python:\n```python\ndef is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\ndef sum_of_differences(n):\n    total_sum = 0\n    for i in range(1, n+1):\n        if is_prime(i) and is_prime(2*n-i):\n            total_sum += abs(2*n - 2*i)\n    return total_sum\n\n# Example usage\nn = 3\nresult = sum_of_differences(n)\nprint(result)  # Output: 6\n```\n\nIn this solution, we define a function `is_prime` to check if a number is prime. Then, we create a function `sum_of_differences` that iterates through all possible partitions of 2n and adds the differences between the larger and smaller parts if the smaller part is prime. Finally, we use an example usage to demonstrate how to use the `sum_of_differences` function."
    ngram_size = 8
    score = score_pair(query, train_text, ngram_size, es, index_name)

    print(json.dumps({
        "score": score,
    }))


if __name__ == "__main__":
    main()


