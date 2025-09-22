# scripts/predict.py
# =========================================
ART_DIR    = r"artifacts"     # or absolute path if you prefer
MODEL_NAME = "knn"            # "knn" | "decision_tree" | "mlp" | "kmeans" | "transformer"
THRESHOLD = 0.35              # soften for KNN; try 0.35–0.45
TOP_K_FALLBACK = 1            # ensure at least k labels if threshold yields none
TEXTS = [
    """In this lecture I give a pedagogical introduction to inflationary cosmology
with a special focus on the quantum generation of cosmological perturbations.
""",
]
# =========================================

import os, sys
from pathlib import Path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from joblib import load
import numpy as np

def _embed_sbert(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import normalize
    model = SentenceTransformer(model_name)
    X = model.encode(list(texts), batch_size=64, show_progress_bar=False,
                     convert_to_numpy=True, normalize_embeddings=False)
    return normalize(X)

def _binarize_with_fallback(A, kind="proba"):
    """
    A: (n_samples, n_labels) probabilities or scores.
    """
    if kind == "proba":
        y = (A >= THRESHOLD).astype(int)
    else:  # scores
        y = (A > 0).astype(int)

    for i in range(y.shape[0]):
        if y[i].sum() == 0:
            top = np.argsort(-A[i])[:TOP_K_FALLBACK]
            y[i, top] = 1
    return y

def _predict_labels(pipe, texts):
    """
    Try proba -> decision_function -> predict, with threshold & top-k fallback.
    If the pipeline expects vectors (e.g., transformer artifact without SBERT),
    we embed first.
    """
    # 1) try predict_proba
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(texts)
            return _binarize_with_fallback(proba, kind="proba")
        except Exception:
            pass

    # 2) try decision_function
    if hasattr(pipe, "decision_function"):
        try:
            scores = pipe.decision_function(texts)
            if scores.ndim == 1:
                scores = scores[:, None]
            return _binarize_with_fallback(scores, kind="score")
        except Exception:
            pass

    # 3) raw predict (may need embedding)
    try:
        y = pipe.predict(texts)
        # if any empty rows, try to rank via last step if possible
        if hasattr(y, "sum") and (y.sum(axis=1) == 0).any():
            last = getattr(pipe, "steps", [])[-1][1] if hasattr(pipe, "steps") else None
            if last is not None:
                try:
                    if hasattr(last, "predict_proba"):
                        Z = getattr(pipe, "__getitem__", None)
                        Z = pipe[:-1].transform(texts) if Z else texts
                        proba = last.predict_proba(Z)
                        return _binarize_with_fallback(proba, kind="proba")
                    if hasattr(last, "decision_function"):
                        Z = pipe[:-1].transform(texts)
                        scores = last.decision_function(Z)
                        if scores.ndim == 1:
                            scores = scores[:, None]
                        return _binarize_with_fallback(scores, kind="score")
                except Exception:
                    pass
        return y
    except ValueError:
        # Transform texts first (artifact expects vectors)
        X = _embed_sbert(texts)
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X)
            return _binarize_with_fallback(proba, kind="proba")
        if hasattr(pipe, "decision_function"):
            scores = pipe.decision_function(X)
            if scores.ndim == 1:
                scores = scores[:, None]
            return _binarize_with_fallback(scores, kind="score")
        return pipe.predict(X)

def main():
    model_path = Path(ART_DIR) / f"{MODEL_NAME}.joblib"
    mlb_path   = Path(ART_DIR) / f"{MODEL_NAME}.mlb.joblib"
    print("[DEBUG] loading:", model_path.resolve())

    arts = load(model_path)   # FitArtifacts(pipeline=..., mlb=...)
    pipe = arts.pipeline
    mlb  = arts.mlb or (load(mlb_path) if mlb_path.exists() else None)

    y_pred_bin = _predict_labels(pipe, TEXTS)

    if mlb is not None:
        labels = [list(lbls) for lbls in mlb.inverse_transform(y_pred_bin)]
    else:
        labels = [[str(x)] if not isinstance(x, (list, tuple)) else list(map(str, x)) for x in y_pred_bin]

    for t, labs in zip(TEXTS, labels):
        print("\n---")
        print("Text:", t[:120].replace("\n", " ") + ("..." if len(t) > 120 else ""))
        print("Predicted labels:", labs)

if __name__ == "__main__":
    main()
