# scripts/train.py
# =========================================
# Hard-coded parameters
MODEL_NAME       = "knn"  # "knn" | "decision_tree" | "mlp" | "kmeans" | "transformer"
PRUNE_MIN_COUNT  = 2      # drop labels with < count (set 0/None to disable)
SEED             = 42
# =========================================

# Make package importable when run directly
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from joblib import dump

# Paths
CSV_PATH  = os.path.join(PROJECT_ROOT, "paper_classification", "data", "preprocess", "arxiv_clean_sample-20k.csv")
ART_DIR   = os.path.join(PROJECT_ROOT, "artifacts")
Path(ART_DIR).mkdir(parents=True, exist_ok=True)

from paper_classification.data.models.factory import ModelFactory

def _parse_labels(raw):
    import re
    if raw is None:
        return []
    toks = re.split(r"[,\;/\|]|\s+", str(raw).strip())
    toks = [t.strip().lower() for t in toks if t and t.strip()]
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _prepare_xy(df: pd.DataFrame, text_col: str, label_col: str, prune_min_count: int | None):
    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    X_text = df[text_col].astype(str)
    y_list = df[label_col].apply(_parse_labels)

    keep = y_list.apply(len) > 0
    if (~keep).any():
        X_text = X_text[keep].reset_index(drop=True)
        y_list = y_list[keep].reset_index(drop=True)

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_list)

    if prune_min_count and prune_min_count > 0:
        counts = pd.Series(Y.sum(axis=0), index=mlb.classes_)
        rare = counts[counts < prune_min_count].index
        if len(rare) > 0:
            import numpy as np
            keep_mask = ~np.isin(mlb.classes_, rare)
            Y = Y[:, keep_mask]
            mlb.classes_ = mlb.classes_[keep_mask]
            row_keep = Y.sum(axis=1) > 0
            X_text = X_text[row_keep].reset_index(drop=True)
            Y = Y[row_keep]

    return X_text.tolist(), Y, mlb

def main():
    TEXT_COL, LABEL_COL = "text_clean", "label"
    df = pd.read_csv(CSV_PATH)
    assert TEXT_COL in df.columns and LABEL_COL in df.columns, \
        f"CSV must contain '{TEXT_COL}' and '{LABEL_COL}'"

    X_text, Y, mlb = _prepare_xy(df, TEXT_COL, LABEL_COL, PRUNE_MIN_COUNT)

    print(f"[INFO] Samples: {len(X_text)} | Labels: {len(mlb.classes_)}")
    print("[INFO] Top labels:\n", pd.Series(Y.sum(axis=0), index=mlb.classes_).sort_values(ascending=False).head(10))

    model = ModelFactory.create(MODEL_NAME)

    # For KMeans, pass labels so it can map cluster -> most frequent label'
    labels_for_kmeans = mlb.inverse_transform(Y)
    if MODEL_NAME.lower() == "kmeans":
        model.fit(X_text, labels=labels_for_kmeans)
    else:
        model.fit(X_text, labels=labels_for_kmeans)

    model_path = Path(ART_DIR) / f"{MODEL_NAME}.joblib"
    mlb_path   = Path(ART_DIR) / f"{MODEL_NAME}.mlb.joblib"
    saved_path = model.save(model_path)
    dump(mlb, mlb_path)

    print(f"[OK] Saved model -> {saved_path}")
    print(f"[OK] Saved label binarizer -> {mlb_path}")

if __name__ == "__main__":
    main()
