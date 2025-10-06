import os
from pathlib import Path
import argparse
import json
import numpy as np
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from utils.path import model_artifact, label_binarizer_path
import sys

try:
    from models.base.base import SBERTVectorizer as _SBERTVectorizer
    sys.modules.setdefault('ensemble', sys.modules[__name__])
    setattr(sys.modules['ensemble'], 'SBERTVectorizer', _SBERTVectorizer)
except Exception:
    pass

try:
    from models.k_means import _KMeansMajority as __KMeansMajority
    sys.modules.setdefault('ensemble', sys.modules[__name__])
    setattr(sys.modules['ensemble'], '_KMeansMajority', __KMeansMajority)
except Exception:
    pass

def _cpu_safe_joblib_load(path):
    import torch
    import torch.serialization as ts
    from joblib import load as _jl_load

    _orig_ts_load = ts.load
    _orig_torch_load = torch.load

    def _ts_load_cpu(f, *args, **kwargs):
        kwargs.setdefault('map_location', 'cpu')
        return _orig_ts_load(f, *args, **kwargs)

    def _torch_load_cpu(f, *args, **kwargs):
        kwargs.setdefault('map_location', 'cpu')
        return _orig_torch_load(f, *args, **kwargs)

    ts.load = _ts_load_cpu
    torch.load = _torch_load_cpu

    try:
        return _jl_load(path)
    finally:
        ts.load = _orig_ts_load
        torch.load = _orig_torch_load

ROOT = Path(__file__).resolve().parents[1]

def read_texts_any(path):
    p = Path(path)
    if p.suffix.lower() in {".txt", ".md"}:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f]
        texts = [ln for ln in lines if ln]
        return texts, None

    try:
        df = pd.read_csv(p, encoding="utf-8")
    except Exception:
        df = pd.read_csv(p, sep=None, engine="python", encoding="utf-8")

    if "text_clean" in df.columns:
        texts = df["text_clean"].astype(str).fillna("").tolist()
        labels = df["label"].astype(str).fillna("").tolist() if "label" in df.columns else None
        return texts, labels

    if df.shape[1] == 1:
        col = df.columns[0]
        return df[col].astype(str).fillna("").tolist(), None

    return [p.read_text(encoding="utf-8", errors="ignore")], None

def setup_paths():
    art_dir = (
        ROOT
        / "Apply-Machine-Learning-to-Classify-Scientific-Paper-Based-on-Abstracts-"
        / "clean_data"
        / "artifacts"
    )

    csv_out_dir = (
        ROOT
        / "Apply-Machine-Learning-to-Classify-Scientific-Paper-Based-on-Abstracts-"
        / "clean_data"
        / "splitted_data"
    )

    return art_dir, csv_out_dir

MODELS = {
    "knn": "models.knn.KNNClassifier",
    "decision_tree": "models.decision_tree.DecisionTreeClassifier", 
    "kmeans": "models.k_means.KMeansClassifier",
    "transformer": "models.transformer.TransformerModel",
}

DEFAULT_THRESHOLD = 0.35
DEFAULT_TOP_K_FALLBACK = 1

def _binarize_with_fallback(A, threshold, top_k, kind="proba"):
    if kind == "proba":
        y = (A >= threshold).astype(int)
    else:
        y = (A > 0).astype(int)

    for i in range(y.shape[0]):
        if y[i].sum() == 0:
            top_indices = np.argsort(-A[i])[:top_k]
            y[i, top_indices] = 1

    return y

def _predict_proba_single_model(pipe, texts, mlb, threshold, top_k):
    n_classes = len(mlb.classes_) if mlb is not None else 1

    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(texts)

            if isinstance(proba, list):
                if len(proba) == 2:
                    proba = np.column_stack([1 - proba[1], proba[1]])
                else:
                    proba = np.column_stack(proba)

            if proba.ndim == 1:
                proba = proba.reshape(-1, 1)

            if proba.shape[1] != n_classes:
                if proba.shape[1] < n_classes:
                    padding = np.zeros((proba.shape[0], n_classes - proba.shape[1]))
                    proba = np.hstack([proba, padding])
                else:
                    proba = proba[:, :n_classes]

            return proba
        except Exception:
            pass

    if hasattr(pipe, "decision_function"):
        try:
            scores = pipe.decision_function(texts)

            if scores.ndim == 1:
                if n_classes == 2:
                    scores = np.column_stack([-scores, scores])
                else:
                    scores = scores.reshape(-1, 1)

            if scores.shape[1] != n_classes:
                if scores.shape[1] < n_classes:
                    padding = np.zeros((scores.shape[0], n_classes - scores.shape[1]))
                    scores = np.hstack([scores, padding])
                else:
                    scores = scores[:, :n_classes]

            from scipy.special import expit
            proba = expit(scores)
            return proba
        except Exception:
            pass

    try:
        y_pred = pipe.predict(texts)

        if isinstance(y_pred, list):
            y_pred = np.asarray(y_pred)

        if y_pred.ndim == 1:
            proba = np.zeros((len(y_pred), n_classes))
            for i, class_idx in enumerate(y_pred):
                if 0 <= int(class_idx) < n_classes:
                    proba[i, int(class_idx)] = 1.0
                else:
                    proba[i, 0] = 1.0
            return proba
        elif y_pred.ndim == 2:
            if y_pred.shape[1] != n_classes:
                if y_pred.shape[1] < n_classes:
                    padding = np.zeros((y_pred.shape[0], n_classes - y_pred.shape[1]))
                    y_pred = np.hstack([y_pred, padding])
                else:
                    y_pred = y_pred[:, :n_classes]
            return y_pred.astype(float)
    except Exception:
        pass

    return np.full((len(texts), n_classes), 1.0 / n_classes)

def load_all_models(art_dir):
    available_models = {}

    for model_name in MODELS.keys():
        model_path = art_dir / f"{model_name}.joblib"
        if model_path.exists():
            try:
                arts = _cpu_safe_joblib_load(model_path)
                if hasattr(arts, "pipeline"):
                    pipe = arts.pipeline
                    mlb = getattr(arts, "mlb", None)
                else:
                    pipe = arts
                    mlb = None

                available_models[model_name] = {
                    'pipeline': pipe,
                    'mlb': mlb
                }
            except Exception:
                pass

    return available_models

def ensemble_predict(models_dict, texts, mlb, threshold, top_k, voting_type="soft"):
    n_samples = len(texts)
    n_classes = len(mlb.classes_) if mlb is not None else 1

    if voting_type == "soft":
        all_probas = []

        for model_name, model_data in models_dict.items():
            pipe = model_data['pipeline']
            model_mlb = model_data['mlb'] or mlb

            proba = _predict_proba_single_model(pipe, texts, model_mlb, threshold, top_k)
            all_probas.append(proba)

        avg_proba = np.mean(all_probas, axis=0)
        y_ensemble = _binarize_with_fallback(avg_proba, threshold, top_k, kind="proba")

        return y_ensemble, avg_proba

    else:
        all_predictions = []

        for model_name, model_data in models_dict.items():
            pipe = model_data['pipeline']
            model_mlb = model_data['mlb'] or mlb

            proba = _predict_proba_single_model(pipe, texts, model_mlb, threshold, top_k)
            y_pred = _binarize_with_fallback(proba, threshold, top_k, kind="proba")
            all_predictions.append(y_pred)

        stacked_preds = np.stack(all_predictions, axis=0)
        y_ensemble = (np.sum(stacked_preds, axis=0) > len(models_dict) / 2).astype(int)

        for i in range(y_ensemble.shape[0]):
            if y_ensemble[i].sum() == 0:
                avg_proba_sample = np.mean([pred[i] for pred in all_predictions], axis=0)
                top_indices = np.argsort(-avg_proba_sample)[:top_k]
                y_ensemble[i, top_indices] = 1

        return y_ensemble, y_ensemble.astype(float)

def compute_metrics(y_true, y_pred, average="macro"):
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    try:
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)

        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

        if y_true.ndim == 2 and y_pred.ndim == 2:
            exact_match_acc = np.all(y_true == y_pred, axis=1).mean()
            prec = precision_score(y_true, y_pred, average=average, zero_division=0)
            rec = recall_score(y_true, y_pred, average=average, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

            return {
                "accuracy": float(exact_match_acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1)
            }
        else:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average=average, zero_division=0)
            rec = recall_score(y_true, y_pred, average=average, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

            return {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1)
            }
    except Exception:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

def plot_ensemble_results(metrics_dict, model_names, voting_type):
    keys = ["accuracy", "precision", "recall", "f1"]

    all_metrics = {}
    for key in keys:
        all_metrics[key] = [metrics_dict[model][key] for model in model_names]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

    for i, metric in enumerate(keys):
        ax = axes[i]
        bars = ax.bar(model_names, all_metrics[metric], color=colors, alpha=0.8, edgecolor='black')

        for bar, val in zip(bars, all_metrics[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylim(0, 1.05)
        ax.set_title(f'{metric.capitalize()} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle(f'Model Comparison - Ensemble ({voting_type.capitalize()} Voting)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

def parse_space_separated_labels(label_string):
    if pd.isna(label_string) or label_string == "" or label_string is None:
        return []

    labels = [label.strip() for label in str(label_string).split(" ") if label.strip()]
    return labels

def main():
    parser = argparse.ArgumentParser(description="Ensemble prediction from multiple trained models.")
    parser.add_argument("--voting", type=str, choices=["soft", "hard"], default="soft",
                       help="Voting mechanism: soft (probability averaging) or hard (majority voting)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                       help="Threshold for binarizing probabilities")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K_FALLBACK,
                       help="Top-K fallback for empty predictions")
    parser.add_argument("--texts_file", type=str, default="val_split.csv",
                       help="Input data file")

    args = parser.parse_args()

    art_dir, csv_out_dir = setup_paths()
    test_path = Path(csv_out_dir) / args.texts_file
    mlb_path = label_binarizer_path()
    config_path = Path(art_dir) / "split_config.json"

    if not test_path.exists():
        return

    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception:
            pass

    if mlb_path.exists():
        try:
            mlb = load(mlb_path)
        except Exception:
            return
    else:
        return

    models_dict = load_all_models(art_dir)

    if not models_dict:
        return

    try:
        texts, labels = read_texts_any(test_path)
    except Exception:
        return

    try:
        y_ensemble, ensemble_proba = ensemble_predict(
            models_dict, texts, mlb, args.threshold, args.top_k, args.voting
        )
    except Exception:
        return

    try:
        ensemble_labels = [list(label_tuple) for label_tuple in mlb.inverse_transform(y_ensemble)]
    except Exception:
        ensemble_labels = [["unknown"] for _ in range(len(y_ensemble))]

    if labels is not None:
        try:
            y_true_labels = [parse_space_separated_labels(s) for s in labels]

            Y_true = mlb.transform(y_true_labels)
            Y_ensemble = y_ensemble

            if Y_true.shape != Y_ensemble.shape:
                min_rows = min(Y_true.shape[0], Y_ensemble.shape[0])
                min_cols = min(Y_true.shape[1], Y_ensemble.shape[1])
                Y_true = Y_true[:min_rows, :min_cols]
                Y_ensemble = Y_ensemble[:min_rows, :min_cols]

            ensemble_scores = compute_metrics(Y_true, Y_ensemble, average="macro")

            all_metrics = {"ensemble": ensemble_scores}

            for model_name, model_data in models_dict.items():
                pipe = model_data['pipeline']
                model_mlb = model_data['mlb'] or mlb

                model_proba = _predict_proba_single_model(pipe, texts, model_mlb, args.threshold, args.top_k)
                y_model = _binarize_with_fallback(model_proba, args.threshold, args.top_k, kind="proba")

                if y_model.shape != Y_true.shape:
                    min_rows = min(Y_true.shape[0], y_model.shape[0])
                    min_cols = min(Y_true.shape[1], y_model.shape[1])
                    y_model_trimmed = y_model[:min_rows, :min_cols]
                else:
                    y_model_trimmed = y_model

                model_scores = compute_metrics(Y_true, y_model_trimmed, average="macro")
                all_metrics[model_name] = model_scores

            print(f"\nFINAL RESULTS COMPARISON:")
            print("="*60)

            model_names = ["ensemble"] + list(models_dict.keys())

            print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
            print("-" * 60)

            for model_name in model_names:
                scores = all_metrics[model_name]
                print(f"{model_name:<15} {scores['accuracy']:<10.4f} {scores['precision']:<10.4f} "
                      f"{scores['recall']:<10.4f} {scores['f1']:<10.4f}")

            print(f"\nParameters: voting={args.voting}, threshold={args.threshold:.2f}, top_k={args.top_k}")

            plot_ensemble_results(all_metrics, model_names, args.voting)

        except Exception:
            pass

if __name__ == "__main__":
    main()
