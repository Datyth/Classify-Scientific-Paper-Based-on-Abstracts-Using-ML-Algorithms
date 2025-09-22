# scripts/predict.py
# =========================================
ART_DIR    = r"data/artifacts"     # or absolute path if you prefer
MODEL_NAME = "knn"            # "knn" | "decision_tree" | "kmeans" 
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
    import numpy as np
    if kind == "proba":
        y = (A >= THRESHOLD).astype(int)
    else:  # "score"
        y = (A > 0).astype(int)
    # ensure at least TOP_K_FALLBACK positives per row
    for i in range(y.shape[0]):
        if y[i].sum() == 0:
            top = np.argsort(-A[i])[:TOP_K_FALLBACK]
            y[i, top] = 1
    return y


def _predict_labels(pipe, texts, mlb):
    """
    Robust prediction:
    - For supervised pipelines (OvR): try predict_proba / decision_function with fallback.
    - For KMeans: return cluster ids (1D), or map yourself if you saved a mapping.
    """
    # 1) Try end-to-end proba
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(texts)
            if proba.ndim == 1:  # single label edge case
                proba = proba[:, None]
            return _binarize_with_fallback(proba, kind="proba")
        except Exception:
            pass

    # 2) Try end-to-end decision_function
    if hasattr(pipe, "decision_function"):
        try:
            scores = pipe.decision_function(texts)
            if scores.ndim == 1:
                scores = scores[:, None]
            return _binarize_with_fallback(scores, kind="score")
        except Exception:
            pass

    # 3) Raw predict (may be 2D for multilabel OvR, or 1D for KMeans)
    try:
        y = pipe.predict(texts)
        import numpy as np
        if isinstance(y, list):
            y = np.asarray(y)
        # KMeans / single-label style → 1D
        if y.ndim == 1:
            return y  # caller will handle (clusters → strings)
        # Multilabel matrix (2D)
        # If any empty rows, try to get ranking from last step only (no re-embedding)
        if (y.sum(axis=1) == 0).any():
            last = pipe.steps[-1][1] if hasattr(pipe, "steps") and pipe.steps else None
            if last is not None:
                Z = pipe[:-1].transform(texts)  # embeddings only
                if hasattr(last, "predict_proba"):
                    proba = last.predict_proba(Z)
                    if proba.ndim == 1:
                        proba = proba[:, None]
                    return _binarize_with_fallback(proba, kind="proba")
                if hasattr(last, "decision_function"):
                    scores = last.decision_function(Z)
                    if scores.ndim == 1:
                        scores = scores[:, None]
                    return _binarize_with_fallback(scores, kind="score")
        return y
    except ValueError:
        # Pipeline expects vectors; embed once, then use the last estimator ONLY.
        Z = pipe[:-1].transform(texts)
        last = pipe.steps[-1][1]
        if hasattr(last, "predict_proba"):
            proba = last.predict_proba(Z)
            if proba.ndim == 1:
                proba = proba[:, None]
            return _binarize_with_fallback(proba, kind="proba")
        if hasattr(last, "decision_function"):
            scores = last.decision_function(Z)
            if scores.ndim == 1:
                scores = scores[:, None]
            return _binarize_with_fallback(scores, kind="score")
        return last.predict(Z)


def main():
    model_path = Path(ART_DIR) / f"{MODEL_NAME}.joblib"
    mlb_path   = Path(ART_DIR) / f"{MODEL_NAME}.mlb.joblib"
    print("[DEBUG] loading:", model_path.resolve())

    arts = load(model_path)   # FitArtifacts(pipeline=..., mlb=...)
    pipe = arts.pipeline
    mlb  = arts.mlb or (load(mlb_path) if mlb_path.exists() else None)

    y_pred = _predict_labels(pipe, TEXTS,mlb)

    labels_out = []
    if mlb is not None:
        # supervised multilabel → matrix or 1D (rare)
        import numpy as np
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 1:
            # single-label ints → map to class names by argmax-like one-hot
            y_bin = np.zeros((y_pred.shape[0], len(mlb.classes_)), dtype=int)
            for i, j in enumerate(y_pred.astype(int)):
                if 0 <= j < len(mlb.classes_):
                    y_bin[i, j] = 1
            y_pred = y_bin
        labels_out = [list(l) for l in mlb.inverse_transform(y_pred)]
    else:
        # unsupervised (e.g., KMeans) → 1D cluster ids
        labels_out = [[str(x)] for x in (y_pred.tolist() if hasattr(y_pred, "tolist") else y_pred)]

    for t, labs in zip(TEXTS, labels_out):
        print("\n---")
        print("Text:", t[:120].replace("\n", " ") + ("..." if len(t) > 120 else ""))
        print("Predicted labels:", labs)

if __name__ == "__main__":
    main()
