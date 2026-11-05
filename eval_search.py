import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
MODEL_NAME = "all-MiniLM-L6-v2"
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def tokens(text):
    return {
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if token not in ENGLISH_STOP_WORDS
    }


def ranked_ids(scores, notes):
    indexes = np.argsort(scores)[::-1][:3]
    return [notes[index]["id"] for index in indexes]


def hit_rate(results, queries):
    if not queries:
        return 0.0
    hits = sum(bool(set(result) & set(query["expected"])) for result, query in zip(results, queries))
    return hits / len(queries)


def main():
    with (ROOT / "data" / "starter_notes.json").open(encoding="utf-8") as file:
        notes = json.load(file)
    with (ROOT / "data" / "search_queries.json").open(encoding="utf-8") as file:
        queries = json.load(file)

    texts = [note["text"] for note in notes]
    vectorizer = TfidfVectorizer()
    note_vectors = vectorizer.fit_transform(texts)
    embedder = SentenceTransformer(MODEL_NAME)
    note_embeddings = embedder.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    )

    approaches = {
        "Keyword overlap": [],
        "TF-IDF cosine": [],
        "Embeddings cosine": [],
    }
    failures = {name: [] for name in approaches}

    for query in queries:
        query_tokens = tokens(query["query"])
        keyword_scores = np.array(
            [
                len(query_tokens & tokens(note["text"])) / max(len(query_tokens), 1)
                for note in notes
            ]
        )
        tfidf_scores = cosine_similarity(
            vectorizer.transform([query["query"]]), note_vectors
        ).ravel()
        query_embedding = embedder.encode(
            [query["query"]], convert_to_numpy=True, normalize_embeddings=True
        )
        embedding_scores = cosine_similarity(query_embedding, note_embeddings).ravel()

        for name, scores in (
            ("Keyword overlap", keyword_scores),
            ("TF-IDF cosine", tfidf_scores),
            ("Embeddings cosine", embedding_scores),
        ):
            results = ranked_ids(scores, notes)
            approaches[name].append(results)
            if not set(results) & set(query["expected"]):
                failures[name].append((query, results))

    no_overlap_queries = [query for query in queries if query["no_overlap"]]
    no_overlap_indexes = [index for index, query in enumerate(queries) if query["no_overlap"]]
    print("Search evaluation (hit rate @3):")
    for name, results in approaches.items():
        all_rate = hit_rate(results, queries)
        no_overlap_results = [results[index] for index in no_overlap_indexes]
        no_overlap_rate = hit_rate(no_overlap_results, no_overlap_queries)
        print(
            f"{name}: all queries {all_rate:.1%} ({sum(bool(set(result) & set(query['expected'])) for result, query in zip(results, queries))}/{len(queries)}); "
            f"no-overlap {no_overlap_rate:.1%} ({sum(bool(set(result) & set(query['expected'])) for result, query in zip(no_overlap_results, no_overlap_queries))}/{len(no_overlap_queries)})"
        )

    print("\nFailed queries (top 3 result IDs):")
    any_failures = False
    for name, failed_queries in failures.items():
        for query, results in failed_queries:
            any_failures = True
            print(
                f"- {name} | query={query['query']!r} | "
                f"expected={query['expected']} | top3={results}"
            )
    if not any_failures:
        print("None")


if __name__ == "__main__":
    main()
