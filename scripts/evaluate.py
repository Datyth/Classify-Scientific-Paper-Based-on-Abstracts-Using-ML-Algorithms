# scripts/evaluate.py
# =========================================
ART_DIR     = r"data/artifacts"
MODEL_NAME  = "decision_tree"   # "knn" | "decision_tree" | "mlp" | "kmeans" |
CSV_PATH    = r"data/csv/val_split.csv"
TEXT_COL    = "text_clean"
LABEL_COL   = "label"
CATEGORIES_TO_SELECT = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']

# Prediction knobs
THRESHOLD   = 0.35     # used when we have probabilities
TOP_K_FALLBACK = 1     # ensure at least k labels per sample
BATCH_SIZE  = 256      # embed/predict in batches to save RAM
# =========================================

from pathlib import Path
import re
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, classification_report

def _parse_labels(raw):
    if raw is None:
        return []
    toks = re.split(r"[,\;/\|]|\s+", str(raw).strip())
    return [t for t in toks if t]

def _to_toplevel(lbl: str) -> str:
    lbl = lbl.strip().lower()
    return lbl.split(".", 1)[0] if "." in lbl else lbl

def _map_and_filter_labels(labels_list, allowed):
    """Map raw labels to top-level, keep only allowed; drop empties later."""
    allowed = set(a.lower() for a in allowed)
    out = []
    for labs in labels_list:
        mapped = list({ _to_toplevel(l) for l in labs })
        kept   = [l for l in mapped if l in allowed]
        out.append(kept)
    return out

def _binarize_with_fallback(A, threshold=0.5, top_k=1, kind="proba"):
    if kind == "proba":
        y = (A >= threshold).astype(int)
    else:
        y = (A > 0).astype(int)
    for i in range(y.shape[0]):
        if y[i].sum() == 0:
            top = np.argsort(-A[i])[:top_k]
            y[i, top] = 1
    return y

def _predict_batch(pipe, texts_batch):
    """Return either binary matrix (supervised) or 1D clusters (KMeans)."""
    # Try full pipeline proba
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(texts_batch)
            if proba.ndim == 1: proba = proba[:, None]
            return _binarize_with_fallback(proba, threshold=THRESHOLD, top_k=TOP_K_FALLBACK, kind="proba")
        except Exception:
            pass
    # Try decision function
    if hasattr(pipe, "decision_function"):
        try:
            scores = pipe.decision_function(texts_batch)
            if scores.ndim == 1: scores = scores[:, None]
            return _binarize_with_fallback(scores, kind="score", top_k=TOP_K_FALLBACK)
        except Exception:
            pass
    # Fallback: raw predict (may be 1D for KMeans)
    return pipe.predict(texts_batch)

def main():
    # Load artifacts
    model_path = Path(ART_DIR) / f"{MODEL_NAME}.joblib"
    mlb_path   = Path(ART_DIR) / f"{MODEL_NAME}.mlb.joblib"
    print("[DEBUG] loading:", model_path.resolve())

    arts = load(model_path)     # FitArtifacts(pipeline=..., mlb=...)
    pipe = arts.pipeline
    mlb  = arts.mlb or (load(mlb_path) if mlb_path.exists() else None)

    # Load data
    df = pd.read_csv(CSV_PATH).dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    # Extract the series of text documents as strings.  Avoid the opaque
    # ``X_text`` name so downstream code is self‑documenting.
    text_series = df[TEXT_COL].astype(str)
    # Parse the raw label strings into lists of tokens using the helper.
    raw_labels = df[LABEL_COL].apply(_parse_labels)

    # Map labels to top‑level and keep only the allowed categories.
    mapped_labels = _map_and_filter_labels(raw_labels, CATEGORIES_TO_SELECT)
    # Keep only rows that have at least one allowed category.  Use a
    # descriptive mask name rather than the generic ``keep``.
    keep_mask = pd.Series([len(l) > 0 for l in mapped_labels])
    text_series = text_series[keep_mask].reset_index(drop=True)
    mapped_labels = [y for i, y in enumerate(mapped_labels) if keep_mask.iloc[i]]

    # If no examples remain after filtering, bail out early.
    if len(text_series) == 0:
        print("No samples left after filtering to allowed categories.")
        return

    # Build a fixed ML Binarizer over exactly these 5 classes (sorted for stability)
    # Use the *saved* mlb classes if available and compatible; otherwise enforce our 5.
    allowed_sorted = sorted({c.lower() for c in CATEGORIES_TO_SELECT})
    # If no MultiLabelBinarizer came from the model, build one over the allowed classes.
    if mlb is None:
        mlb = MultiLabelBinarizer(classes=allowed_sorted)
        Y_true = mlb.fit_transform(mapped_labels)
    else:
        # If the trained binarizer classes differ, rebuild it to ensure the
        # evaluation runs on a consistent label set.
        trained_classes = [c.lower() for c in mlb.classes_]
        if sorted(trained_classes) != allowed_sorted:
            mlb = MultiLabelBinarizer(classes=allowed_sorted)
        Y_true = mlb.fit_transform(mapped_labels)

    # Predict in batches
    preds = []
    n = len(text_series)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = list(text_series.iloc[start:end])
        yb = _predict_batch(pipe, batch)
        if not isinstance(yb, np.ndarray):
            yb = np.asarray(yb)
        if yb.ndim == 1:
            yb = np.zeros((yb.shape[0], len(mlb.classes_)), dtype=int)
        preds.append(yb)

    Y_pred = np.vstack(preds)

    # Metrics
    print(f"\n=== Evaluation on {n} samples ===")
    print("F1-micro :", round(f1_score(Y_true, Y_pred, average="micro", zero_division=0), 4))
    print("F1-macro :", round(f1_score(Y_true, Y_pred, average="macro", zero_division=0), 4))
    print("Subset accuracy (exact match):", round(accuracy_score(Y_true, Y_pred), 4))
    print("Hamming loss:", round(hamming_loss(Y_true, Y_pred), 4))
    print("\nPer-class report:\n", classification_report(Y_true, Y_pred, target_names=list(mlb.classes_), zero_division=0))

if __name__ == "__main__":
    main()
