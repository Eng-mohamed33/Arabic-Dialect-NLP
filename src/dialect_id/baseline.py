"""Character n-gram TF-IDF + LinearSVC baseline."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data.text import normalize_arabic


def build_model() -> Pipeline:
    return Pipeline([("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 5), sublinear_tf=True)),
                     ("classifier", LinearSVC())])


def build_nb_model() -> Pipeline:
    """Naive Bayes comparison baseline using the same character features."""
    return Pipeline([("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 5), sublinear_tf=True)),
                     ("classifier", MultinomialNB())])


def train(frame: pd.DataFrame, *, model_type: str = "linearsvc") -> Pipeline:
    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    if model_type == "linearsvc":
        model = build_model()
    elif model_type == "naive_bayes":
        model = build_nb_model()
    else:
        raise ValueError("model_type must be 'linearsvc' or 'naive_bayes'")
    model.fit(frame["text"].map(normalize_arabic), frame["label"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model = train(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(f"saved model to {args.output}")


if __name__ == "__main__":
    main()

