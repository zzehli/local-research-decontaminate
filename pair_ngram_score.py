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
        #  analyzer="tulu_analyzer"
        resp = es.indices.analyze(index=index_name, text=text)
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
    query = "3 loaves of bread cost 3 * $2 = $<<3*2=6>>6.\n2 bagels cost 2 * $1 = $<<2*1=2>>2.\nThe loaves of bread cost $6 - $2 = $<<6-2=4>>4 more than the bagels.\n#### 4"
    train_text = "To solve this problem, we need to find a function that generates an infinite sequence of integers where each integer can be expressed as the sum of the squares of two positive integers. We are given the first three members of the sequence: 72, 288, and 800. We need to verify if the proposed function \\( f(n) = 2(n+1)^2(n+2)^2 \\) generates these numbers and find the next three members of the sequence.\n\n1. **Verify the given function for the first three members:**\n\n   Let's check if \\( f(n) = 2(n+1)^2(n+2)^2 \\) generates the numbers 72, 288, and 800.\n\n   - For \\( n = 1 \\):\n     \\[\n     f(1) = 2(1+1)^2(1+2)^2 = 2 \\cdot 2^2 \\cdot 3^2 = 2 \\cdot 4 \\cdot 9 = 72\n     \\]\n     This matches the first number in the sequence.\n\n   - For \\( n = 2 \\):\n     \\[\n     f(2) = 2(2+1)^2(2+2)^2 = 2 \\cdot 3^2 \\cdot 4^2 = 2 \\cdot 9 \\cdot 16 = 288\n     \\]\n     This matches the second number in the sequence.\n\n   - For \\( n = 3 \\):\n     \\[\n     f(3) = 2(3+1)^2(3+2)^2 = 2 \\cdot 4^2 \\cdot 5^2 = 2 \\cdot 16 \\cdot 25 = 800\n     \\]\n     This matches the third number in the sequence.\n\n2. **Find the next three members of the sequence:**\n\n   Using the function \\( f(n) = 2(n+1)^2(n+2)^2 \\), we can find the next three members by evaluating the function for \\( n = 4, 5, \\) and \\( 6 \\).\n\n   - For \\( n = 4 \\):\n     \\[\n     f(4) = 2(4+1)^2(4+2)^2 = 2 \\cdot 5^2 \\cdot 6^2 = 2 \\cdot 25 \\cdot 36 = 1800\n     \\]\n\n   - For \\( n = 5 \\):\n     \\[\n     f(5) = 2(5+1)^2(5+2)^2 = 2 \\cdot 6^2 \\cdot 7^2 = 2 \\cdot 36 \\cdot 49 = 3528\n     \\]\n\n   - For \\( n = 6 \\):\n     \\[\n     f(6) = 2(6+1)^2(6+2)^2 = 2 \\cdot 7^2 \\cdot 8^2 = 2 \\cdot 49 \\cdot 64 = 6272\n     \\]\n\n3. **Conclusion:**\n\n   The function \\( f(n) = 2(n+1)^2(n+2)^2 \\) correctly generates the sequence of integers where each integer can be expressed as the sum of the squares of two positive integers. The next three members of the sequence are 1800, 3528, and 6272.\n\nThe final answer is \\( \\boxed{ 1800, 3528, 6272 } \\)"
    ngram_size = 8
    score = score_pair(query, train_text, ngram_size, es, index_name)

    print(json.dumps({
        "score": score,
    }))


if __name__ == "__main__":
    main()


