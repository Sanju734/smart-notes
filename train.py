from pathlib import Path
from importlib.metadata import version
import re
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline


ROOT = Path(__file__).resolve().parent
CS = (1, 3, 10, 30, 100)
CV_SPLITS = 5
RANDOM_SEED = 42
PROB_THRESHOLD = 0.45
COVERAGE_THRESHOLD = 0.20
PROB_THRESHOLDS = (0.40, 0.45, 0.50)
COVERAGE_THRESHOLDS = (0.01, 0.10, 0.20, 0.25, 0.34)
WORD_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def load_csv(filename):
    data = pd.read_csv(ROOT / "data" / filename)
    if "text" not in data.columns:
        raise ValueError(f"{filename} must contain a text column")
    return data


def load_labeled_csv(filename):
    data = load_csv(filename)
    if "category" not in data.columns:
        raise ValueError(f"{filename} must contain a category column")
    return data


def make_model(c, use_char_features=False):
    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    if use_char_features:
        features = FeatureUnion(
            [
                ("word", word_vectorizer),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))),
            ]
        )
    else:
        features = word_vectorizer

    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    C=c, random_state=RANDOM_SEED, max_iter=1000, solver="liblinear"
                ),
            ),
        ]
    )


def make_v1_model():
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            (
                "classifier",
                LogisticRegression(
                    random_state=RANDOM_SEED, max_iter=1000, solver="liblinear"
                ),
            ),
        ]
    )


def vocabulary_coverage(text, vocabulary):
    tokens = [
        token
        for token in WORD_PATTERN.findall(text.lower())
        if token not in ENGLISH_STOP_WORDS
    ]
    if not tokens:
        return 0.0
    known_tokens = sum(token in vocabulary for token in tokens)
    return known_tokens / len(tokens)


def coverage_scores(model, texts):
    word_vectorizer = model.named_steps["features"].transformer_list[0][1]
    vocabulary = word_vectorizer.vocabulary_
    return np.array([vocabulary_coverage(text, vocabulary) for text in texts])


def report_metrics(name, actual, predicted, labels):
    print(f"\n{name}")
    print(f"Accuracy: {accuracy_score(actual, predicted):.3f}")
    print(
        "Macro-F1: "
        f"{f1_score(actual, predicted, labels=labels, average='macro', zero_division=0):.3f}"
    )
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(f"Labels: {labels}")
    print(confusion_matrix(actual, predicted, labels=labels))


def print_misclassifications(model, train, validation, predictions, probabilities):
    feature_model = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    feature_names = feature_model.get_feature_names_out()
    train_matrix = feature_model.transform(train["text"])

    print("\nMisclassified validation notes:")
    found_error = False
    for index, (text, actual, predicted) in enumerate(
        zip(validation["text"], validation["category"], predictions)
    ):
        if actual == predicted:
            continue
        found_error = True
        row = feature_model.transform([text])
        class_index = list(classifier.classes_).index(predicted)
        contributions = row.multiply(classifier.coef_[class_index]).toarray()[0]
        top_features = np.argsort(contributions)[::-1]
        top_features = [i for i in top_features if contributions[i] > 0][:5]
        feature_text = ", ".join(
            f"{feature_names[i]} ({contributions[i]:.3f})" for i in top_features
        )
        if not feature_text:
            feature_text = "no positive feature contributions"

        similarities = cosine_similarity(row, train_matrix).ravel()
        nearest = np.argsort(similarities)[::-1][:3]
        print(
            f"- {text!r} | actual={actual} | predicted={predicted} "
            f"| top_probability={probabilities[index].max():.3f}"
        )
        print(f"  Top features toward {predicted}: {feature_text}")
        print("  Most similar training notes:")
        for train_index in nearest:
            print(
                f"    {train['text'].iloc[train_index]!r} "
                f"| label={train['category'].iloc[train_index]} "
                f"| cosine_similarity={similarities[train_index]:.3f}"
            )
    if not found_error:
        print("None")


def print_threshold_table(
    out_of_scope,
    out_probabilities,
    out_coverages,
    validation,
    validation_probabilities,
    validation_coverages,
):
    print("\nThreshold trade-off (flagged if probability OR coverage is below threshold):")
    print("Prob threshold | Coverage threshold | Out-of-scope flagged | Validation flagged")
    for probability_threshold in PROB_THRESHOLDS:
        for coverage_threshold in COVERAGE_THRESHOLDS:
            out_flagged = (
                (out_probabilities < probability_threshold)
                | (out_coverages < coverage_threshold)
            )
            validation_flagged = (
                (validation_probabilities < probability_threshold)
                | (validation_coverages < coverage_threshold)
            )
            out_count = int(out_flagged.sum())
            validation_count = int(validation_flagged.sum())
            print(
                f"{probability_threshold:>13.2f} | {coverage_threshold:>17.2f} | "
                f"{100 * out_count / len(out_flagged):5.1f}% ({out_count}/"
                f"{len(out_flagged)})       | "
                f"{100 * validation_count / len(validation_flagged):5.1f}% "
                f"({validation_count}/{len(validation_flagged)})"
            )
            unflagged_out_notes = [
                text
                for text, flagged in zip(out_of_scope["text"], out_flagged)
                if not flagged
            ]
            print(f"  Out-of-scope not flagged: {unflagged_out_notes or 'None'}")

            flagged_validation_notes = [
                (
                    text,
                    probability,
                    coverage,
                )
                for text, probability, coverage, flagged in zip(
                    validation["text"],
                    validation_probabilities,
                    validation_coverages,
                    validation_flagged,
                )
                if flagged
            ]
            print("  Flagged validation notes (top probability, coverage):")
            for text, probability, coverage in flagged_validation_notes:
                print(
                    f"    - {text!r} "
                    f"(probability={probability:.3f}, coverage={coverage:.3f})"
                )
            if not flagged_validation_notes:
                print("    None")


def print_out_of_scope_scores(out_of_scope, probabilities, coverages):
    print("\nOut-of-scope note scores:")
    for text, probability, coverage in zip(
        out_of_scope["text"], probabilities, coverages
    ):
        print(
            f"- {text!r} | top_probability={probability:.3f} "
            f"| vocabulary_coverage={coverage:.3f}"
        )


def main():
    train = load_labeled_csv("train.csv")
    validation = load_labeled_csv("validation.csv")
    out_of_scope = load_csv("out_of_scope_examples.csv")
    labels = sorted(train["category"].unique())

    print(f"Training rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"scikit-learn version: {sklearn.__version__}")
    print(f"sentence-transformers version: {version('sentence-transformers')}")

    baseline_label = train["category"].mode().iloc[0]
    baseline_predictions = [baseline_label] * len(validation)
    report_metrics(
        f"Most-frequent-label baseline (label={baseline_label})",
        validation["category"],
        baseline_predictions,
        labels,
    )

    # Keep the original setup as v1 for a direct before/after comparison.
    v1 = make_v1_model()
    v1.fit(train["text"], train["category"])
    v1_predictions = v1.predict(validation["text"])
    report_metrics("V1: original TF-IDF + LogisticRegression", validation["category"], v1_predictions, labels)

    cv = StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED
    )
    configurations = []
    print(f"\n{CV_SPLITS}-fold stratified CV on train.csv only (macro-F1):")
    for c in CS:
        for use_char_features in (False, True):
            name = "word 1-2 grams + char_wb 2-5 grams" if use_char_features else "word 1-2 grams"
            model = make_model(c, use_char_features)
            scores = cross_val_score(
                model,
                train["text"],
                train["category"],
                cv=cv,
                scoring="f1_macro",
            )
            mean_score = scores.mean()
            configurations.append((mean_score, c, use_char_features))
            print(f"C={c:>3}, features={name}: {mean_score:.3f} (std {scores.std():.3f})")

    best_score, best_c, best_uses_char = max(configurations, key=lambda item: item[0])
    best_feature_name = (
        "word 1-2 grams + char_wb 2-5 grams"
        if best_uses_char
        else "word 1-2 grams"
    )
    print(
        f"\nSelected: C={best_c}, features={best_feature_name}, "
        f"CV macro-F1={best_score:.3f}"
    )

    model = make_model(best_c, best_uses_char)
    model.fit(train["text"], train["category"])
    predictions = model.predict(validation["text"])
    report_metrics("Selected model (validation)", validation["category"], predictions, labels)

    validation_probabilities = model.predict_proba(validation["text"])
    print_misclassifications(
        model, train, validation, predictions, validation_probabilities
    )
    out_of_scope_probabilities = model.predict_proba(out_of_scope["text"])
    out_top_probabilities = out_of_scope_probabilities.max(axis=1)
    validation_top_probabilities = validation_probabilities.max(axis=1)
    out_coverages = coverage_scores(model, out_of_scope["text"])
    validation_coverages = coverage_scores(model, validation["text"])
    print_out_of_scope_scores(out_of_scope, out_top_probabilities, out_coverages)
    print_threshold_table(
        out_of_scope,
        out_top_probabilities,
        out_coverages,
        validation,
        validation_top_probabilities,
        validation_coverages,
    )
    print(
        f"\nSaved uncertainty thresholds: probability={PROB_THRESHOLD:.2f}, "
        f"coverage={COVERAGE_THRESHOLD:.2f}"
    )

    model_path = ROOT / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "prob_threshold": PROB_THRESHOLD,
            "coverage_threshold": COVERAGE_THRESHOLD,
        },
        model_path,
    )
    print(f"\nSaved model to {model_path.name}")


if __name__ == "__main__":
    main()
