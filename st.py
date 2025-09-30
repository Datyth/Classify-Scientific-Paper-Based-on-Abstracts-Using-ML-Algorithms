# st.py
from pathlib import Path
import numpy as np
from joblib import load
import streamlit as st
from utils.path import model_artifact, label_binarizer_path
project_root = Path(__file__).resolve().parents
def setup_paths() -> tuple[Path, Path]:
    art_dir = project_root /"clean_data" /"artifacts"
    csv_out_dir = project_root /"clean_data"/ "splits" 
    return art_dir, csv_out_dir

@st.cache_resource
def load_artifacts(model_name: str):
    model_path = model_artifact(model_name)   
    lb_path    = label_binarizer_path()
    obj = load(model_path)
    if hasattr(obj, "pipeline"):
        pipe = obj.pipeline
        model_mlb = getattr(obj, "mlb", None)
    else:
        pipe = obj
        model_mlb = None

    label_binarizer = model_mlb if model_mlb is not None else load(lb_path)
    classes = list(getattr(label_binarizer, "classes_", []))
    return pipe, label_binarizer, classes

def _scores_from_pipeline(pipe, texts):
    if hasattr(pipe, "predict_proba"):
        P = pipe.predict_proba(texts)
        if isinstance(P, list):
            P = np.column_stack([p[:, 1] if p.ndim == 2 else p for p in P])
        return "proba", np.asarray(P)
    if hasattr(pipe, "decision_function"):
        S = pipe.decision_function(texts)
        if isinstance(S, list):
            S = np.column_stack(S)
        return "score", np.asarray(S)
    Y = pipe.predict(texts)
    return "hard", np.asarray(Y)

def multilabel_select(scores, kind: str, threshold: float, top_k: int):
    if kind == "hard":
        arr = np.asarray(scores)
        if arr.ndim == 1:
            n = int(arr.max()) + 1 if arr.size else 0
            out = np.zeros((arr.size, n), dtype=int)
            for i, j in enumerate(arr.astype(int)):
                if 0 <= j < n:
                    out[i, j] = 1
            return out
        return (arr > 0).astype(int)

    X = np.asarray(scores)
    if X.ndim == 1:
        X = X[None, :]
    Y = (X >= threshold).astype(int)
    for i in range(Y.shape[0]):
        if Y[i].sum() == 0 and top_k > 0:
            k = min(top_k, Y.shape[1])
            topk_idx = np.argsort(-X[i])[:k]
            Y[i, topk_idx] = 1
    return Y
st.set_page_config(page_title="Abstract Classifier")
st.title(" Abstract Classifier (3 models of ML)")
model_name = st.selectbox(
    "Model",
    ("kmeans", "knn", "decision_tree"),
    index=0,
    help="Loads artifacts/<model>.joblib"
)
pipe, label_binarizer, classes = load_artifacts(model_name)

abstract = st.text_area("Paste abstract here", height=200, placeholder="Paste one abstract...")

col1, col2, col3 = st.columns(3)
with col1:
    threshold = st.slider("Threshold", 0.0, 1.0, 0.35, 0.05)
with col2:
    top_k = st.number_input(
        "Top-K fallback",
        min_value=0, max_value=5, value=1, step=1,
        help="If nothing passes the threshold, still return the top K labels by score."
    )
with col3:
    run = st.button("Predict", use_container_width=True)

if run:
    text = (abstract or "").strip()
    if not text:
        st.warning("Please paste an abstract first.")
    else:
        kind, scores_or_pred = _scores_from_pipeline(pipe, [text])
        Ybin = multilabel_select(scores_or_pred, kind, threshold, top_k)
        labels = label_binarizer.inverse_transform(Ybin)

        st.subheader("Predicted labels")
        if labels and labels[0]:
            st.success(", ".join(labels[0]))
        else:
            st.info("No label passed the threshold.")
