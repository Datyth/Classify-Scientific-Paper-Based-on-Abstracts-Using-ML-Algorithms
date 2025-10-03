#evaluate.py
import argparse
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    hamming_loss,
    classification_report,
)
import joblib
from models.base.base import FitArtifacts
from models.base.base import SBERTVectorizer

def _parse_labels(raw: str | Sequence[str] | None) -> List[str]:
    """Parse a raw label into tokens."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip().lower() for x in raw if str(x).strip()]
    toks = re.split(r"[\,;/\|]|\s+", str(raw).strip())
    return [t.strip().lower() for t in toks if t.strip()]


def _to_toplevel(label: str) -> str:
    label = label.strip().lower()
    return label.split(".", 1)[0] if "." in label else label


def _filter_labels(
    texts: Iterable[str],
    raw_labels: Iterable[Sequence[str]],
    allowed_cats: Sequence[str],
) -> tuple[list[str], list[list[str]]]:
    """Filter labels to allowed categories (top‑level) and return texts and mapped labels."""
    allowed_set = set(c.lower() for c in allowed_cats)
    filtered_texts: list[str] = []
    mapped_labels: list[list[str]] = []
    for text, labs in zip(texts, raw_labels):
        mapped = { _to_toplevel(l) for l in labs }
        kept = [t for t in mapped if t in allowed_set]
        if kept:
            filtered_texts.append(str(text))
            mapped_labels.append(kept)
    return filtered_texts, mapped_labels


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a validation CSV")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the evaluation CSV")
    parser.add_argument("--model_file", type=str, required=True, help="Path to the trained model file (joblib)")
    parser.add_argument("--text_col", type=str, default="text_clean", help="Name of the text column")
    parser.add_argument("--label_col", type=str, default="label", help="Name of the label column")
    parser.add_argument(
        "--categories", nargs="+", default=["astro-ph", "cond-mat", "cs", "math", "physics"],
        help="Top-level categories used in training",
    )
    args = parser.parse_args(argv)

    # Load model artifacts
    artifacts = joblib.load(args.model_file)
    # Ensure pipeline and mlb exist
    pipeline = artifacts.pipeline
    mlb = artifacts.mlb
    # Allowed classes come from mlb if present; otherwise from args
    if mlb is not None:
        allowed = [c.lower() for c in mlb.classes_]
    else:
        allowed = [c.lower() for c in args.categories]

    # Load evaluation data
    df = pd.read_csv(args.csv_path).dropna(subset=[args.text_col, args.label_col]).reset_index(drop=True)
    texts = df[args.text_col].astype(str).tolist()
    raw_labels = df[args.label_col].apply(_parse_labels).tolist()
    filtered_texts, mapped_labels = _filter_labels(texts, raw_labels, allowed)
    if not filtered_texts:
        print("No samples remain after filtering categories.")
        return 1
    # Convert mapped labels to binary using the same MultiLabelBinarizer
    if mlb is not None:
        Y_true = mlb.transform(mapped_labels)
    else:
        # Build new binarizer if none existed (e.g. KMeans unsupervised)
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer(classes=sorted(allowed))
        Y_true = mlb.fit_transform(mapped_labels)
    # Predict using the trained pipeline
    Y_pred = np.asarray(pipeline.predict(filtered_texts))
    # If predictions are 1D, expand to 2D for metrics
    if Y_pred.ndim == 1:
        Y_pred = Y_pred[:, None]
    # Compute metrics
    print(f"\n=== Evaluation on {len(filtered_texts)} samples ===")
    print("F1-micro :", round(f1_score(Y_true, Y_pred, average="micro", zero_division=0), 4))
    print("F1-macro :", round(f1_score(Y_true, Y_pred, average="macro", zero_division=0), 4))
    print("Subset accuracy (exact match):", round(accuracy_score(Y_true, Y_pred), 4))
    print("Hamming loss:", round(hamming_loss(Y_true, Y_pred), 4))
    # Print classification report per class
    class_names = list(mlb.classes_)
    print("\nPer-class report:\n", classification_report(Y_true, Y_pred, target_names=class_names, zero_division=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())