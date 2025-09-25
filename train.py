# scripts/train.py
# =========================================
# Hard-coded parameters
MODEL_NAME       = "decision_tree"  # "knn" | "decision_tree"  | "kmeans" | "transformer"
PRUNE_MIN_COUNT  = 2      # drop labels with < count (set 0/None to disable)
CATEGORIES_TO_SELECT = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']
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

from sklearn.model_selection import train_test_split

# Paths
CSV_PATH  = os.path.join(PROJECT_ROOT, "Apply-Machine-Learning-to-Classify-Scientific-Paper-Based-on-Abstracts-","data", "csv", "arxiv_clean_sample-5k.csv")
ART_DIR   = os.path.join(PROJECT_ROOT, "Apply-Machine-Learning-to-Classify-Scientific-Paper-Based-on-Abstracts-","data","artifacts")
CSV_OUT_DIR = os.path.join(PROJECT_ROOT, "Apply-Machine-Learning-to-Classify-Scientific-Paper-Based-on-Abstracts-","data", "csv")
Path(ART_DIR).mkdir(parents=True, exist_ok=True)

from models.base.factory import ModelFactory

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
def _to_toplevel(lbl: str) -> str:
    if "." in lbl:
        return lbl.split(".", 1)[0].lower()
    return lbl.lower()
def _prepare_xy_topcats(df: pd.DataFrame, text_col: str, label_col: str, cats: list[str]):
    X_text = df[text_col].astype(str)
    y_raw  = df[label_col].apply(_parse_labels)
    y_top = []
    keep_rows = []
    catset = set(c.lower() for c in cats)
    for i, labels in enumerate(y_raw):
        mapped = list({ _to_toplevel(l) for l in labels })  # dedupe per sample
        filtered = [t for t in mapped if t in catset]
        if filtered:
            y_top.append(filtered)
            keep_rows.append(True)
        else:
            keep_rows.append(False)

    # filter rows
    X_text = X_text[keep_rows].reset_index(drop=True)
    y_top  = pd.Series(y_top)
    mlb = MultiLabelBinarizer(classes=sorted(catset))
    Y = mlb.fit_transform(y_top)

    return X_text.tolist(), Y, mlb


def main():
    TEXT_COL, LABEL_COL = "text_clean", "label"
    df = pd.read_csv(CSV_PATH).dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)

    X_text, Y, mlb = _prepare_xy_topcats(df, TEXT_COL, LABEL_COL, CATEGORIES_TO_SELECT)

    print(f"[INFO] Samples kept: {len(X_text)}")
    print(f"[INFO] Classes: {list(mlb.classes_)}")
    print("[INFO] Label counts:\n", pd.Series(Y.sum(axis=0), index=mlb.classes_).astype(int))

    model = ModelFactory.create(MODEL_NAME)

    # ---- stratified split (multilabel-aware if package is available) ----
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        idx = np.arange(len(X_text))
        (train_idx, val_idx) = next(msss.split(idx, Y))
        print("[INFO] Used MultilabelStratifiedShuffleSplit (80/20).")
    except Exception:
        # Fallback: random split (not stratified but OK if package missing)
        print("[WARN] iterative-stratification not found; falling back to random split (80/20).")
        idx = np.arange(len(X_text))
        train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=SEED, shuffle=True)

    # Build splits
    X_train = [X_text[i] for i in train_idx]
    Y_train = Y[train_idx]
    X_val   = [X_text[i] for i in val_idx]
    Y_val   = Y[val_idx]

    # Fit on TRAIN only (pass labels so kmeans mapper can learn cluster→label)
    labels_train = mlb.inverse_transform(Y_train)
    model.fit(X_train, labels=labels_train)

    # ---- save a validation CSV for evaluate.py ----
    val_path =Path(CSV_OUT_DIR)/"val_split.csv"
    pd.DataFrame({
        "text_clean": X_val,
        "label": [" ".join(labs) for labs in mlb.inverse_transform(Y_val)]
    }).to_csv(val_path, index=False)
    print(f"[OK] Wrote validation split -> {val_path}")

    # ---- save artifacts ----
    model_path = Path(ART_DIR) / f"{MODEL_NAME}.joblib"
    mlb_path   = Path(ART_DIR) / f"{MODEL_NAME}.mlb.joblib"
    saved_path = model.save(model_path)
    dump(mlb, mlb_path, compress=3, protocol=5)

    print(f"[OK] Saved model -> {saved_path}")
    print(f"[OK] Saved label binarizer -> {mlb_path}")


if __name__ == "__main__":
    main()
