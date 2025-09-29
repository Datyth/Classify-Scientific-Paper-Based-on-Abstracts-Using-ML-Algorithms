import os
import sys
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


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def setup_paths():
    art_dir = os.path.join(
        PROJECT_ROOT,
        "Apply-Machine-Learning-to-Classify-Scientific-Paper-Based-on-Abstracts-",
        "clean_data",
        "artifacts",
    )
    csv_out_dir = os.path.join(
        PROJECT_ROOT,
        "Apply-Machine-Learning-to-Classify-Scientific-Paper-Based-on-Abstracts-",
        "clean_data",
        "splitted_data",
    )
    return art_dir, csv_out_dir


MODELS = {
    "knn": "models.knn.KNNClassifier",
    "decision_tree": "models.decision_tree.DecisionTreeClassifier", 
    "kmeans": "models.k_means.KMeansClassifier",
    "transformer": "models.transformer.TransformerModel",
}


DEFAULT_MODEL_NAME = "knn"
DEFAULT_THRESHOLD = 0.35
DEFAULT_TOP_K_FALLBACK = 1


DEFAULT_TEXTS = [
    """In this lecture I give a pedagogical introduction to inflationary cosmology
with a special focus on the quantum generation of cosmological perturbations.
""",
]


def _binarize_with_fallback(A, threshold, top_k, kind="proba"):
    """Convert probability/score matrix to binary predictions with fallback"""
    if kind == "proba":
        y = (A >= threshold).astype(int)
    else:
        y = (A > 0).astype(int)
    
    # Apply fallback for samples with no positive predictions
    for i in range(y.shape[0]):
        if y[i].sum() == 0:
            top_indices = np.argsort(-A[i])[:top_k]
            y[i, top_indices] = 1
    return y


def _predict_labels(pipe, texts, mlb, threshold, top_k):
    """Enhanced prediction function with better error handling for different model types"""
    print(f"Predicting for {len(texts)} texts...")
    
    # Get number of expected classes
    n_classes = len(mlb.classes_) if mlb is not None else 1
    print(f"Expected number of classes: {n_classes}")
    
    # Try predict_proba first (for probabilistic models)
    if hasattr(pipe, "predict_proba"):
        try:
            print("Using predict_proba...")
            proba = pipe.predict_proba(texts)
            
            # Handle different proba formats
            if isinstance(proba, list):  # Some models return list of arrays
                if len(proba) == 2:  # Binary classification case
                    proba = np.column_stack([1 - proba[1], proba[1]])
                else:
                    proba = np.column_stack(proba)
            
            if proba.ndim == 1:
                proba = proba.reshape(-1, 1)
            
            # Ensure correct shape
            if proba.shape[1] != n_classes:
                print(f"Warning: predict_proba returned {proba.shape[1]} classes, expected {n_classes}")
                if proba.shape[1] < n_classes:
                    # Pad with zeros
                    padding = np.zeros((proba.shape[0], n_classes - proba.shape[1]))
                    proba = np.hstack([proba, padding])
                else:
                    # Truncate
                    proba = proba[:, :n_classes]
            
            return _binarize_with_fallback(proba, threshold, top_k, kind="proba")
            
        except Exception as e:
            print(f"predict_proba failed: {e}")
    
    # Try decision_function (for SVM-like models)
    if hasattr(pipe, "decision_function"):
        try:
            print("Using decision_function...")
            scores = pipe.decision_function(texts)
            if scores.ndim == 1:
                if n_classes == 2:  # Binary case
                    scores = np.column_stack([-scores, scores])
                else:
                    scores = scores.reshape(-1, 1)
            
            # Ensure correct shape
            if scores.shape[1] != n_classes:
                if scores.shape[1] < n_classes:
                    padding = np.zeros((scores.shape[0], n_classes - scores.shape[1]))
                    scores = np.hstack([scores, padding])
                else:
                    scores = scores[:, :n_classes]
            
            return _binarize_with_fallback(scores, 0, top_k, kind="score")
            
        except Exception as e:
            print(f"decision_function failed: {e}")
    
    # Try direct predict (for clustering models like KMeans)
    try:
        print("Using predict...")
        y_pred = pipe.predict(texts)
        print(f"Predict returned shape: {np.array(y_pred).shape}")
        
        if isinstance(y_pred, list):
            y_pred = np.asarray(y_pred)
        
        # Handle different prediction formats
        if y_pred.ndim == 1:
            print("Processing single-class predictions...")
            # Single class prediction - convert to one-hot
            y_binary = np.zeros((len(y_pred), n_classes), dtype=int)
            for i, class_idx in enumerate(y_pred):
                if 0 <= int(class_idx) < n_classes:
                    y_binary[i, int(class_idx)] = 1
                else:
                    # Fallback to first class if index out of bounds
                    y_binary[i, 0] = 1
            return y_binary
        
        elif y_pred.ndim == 2:
            print("Processing multi-label predictions...")
            # Multi-label prediction (already binary)
            
            # Ensure correct shape
            if y_pred.shape[1] != n_classes:
                print(f"Warning: Prediction shape {y_pred.shape[1]} != expected {n_classes}")
                if y_pred.shape[1] < n_classes:
                    # Pad with zeros
                    padding = np.zeros((y_pred.shape[0], n_classes - y_pred.shape[1]), dtype=int)
                    y_pred = np.hstack([y_pred, padding])
                else:
                    # Truncate
                    y_pred = y_pred[:, :n_classes]
            
            # Apply fallback for empty predictions
            for i in range(y_pred.shape[0]):
                if y_pred[i].sum() == 0:
                    y_pred[i, 0] = 1  # Assign to first class
                    
            return y_pred.astype(int)
        
        else:
            print(f"Unexpected prediction shape: {y_pred.shape}")
            raise ValueError(f"Unexpected prediction dimensionality: {y_pred.ndim}")
            
    except Exception as e:
        print(f"predict failed: {e}")
        
        # Final fallback
        print("Using fallback predictions...")
        y_fallback = np.zeros((len(texts), n_classes), dtype=int)
        y_fallback[:, 0] = 1  # Assign all to first class
        return y_fallback


def compute_metrics(y_true, y_pred, average="macro"):
    """Compute classification metrics with better error handling"""
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    try:
        # Ensure both arrays have same shape and type
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        
        print(f"Computing metrics - Y_true shape: {y_true.shape}, Y_pred shape: {y_pred.shape}")
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        if y_true.ndim == 2 and y_pred.ndim == 2:
            # Multi-label case - use exact match accuracy
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
            # Multi-class case
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
            
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def plot_single_metric(value, title):
    """Plot a single metric bar chart"""
    plt.figure(figsize=(6, 4))
    bars = plt.bar([title], [value], color="#4e79a7", alpha=0.8, edgecolor='black', linewidth=1)
    plt.ylim(0, 1.05)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'{title.capitalize()}: {value:.4f}', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def plot_overview(metrics_dict, title):
    """Plot overview of all metrics"""
    keys = ["accuracy", "precision", "recall", "f1"]
    vals = [metrics_dict[k] for k in keys]
    colors = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759"]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(keys, vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    plt.ylim(0, 1.05)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Metrics', fontsize=12)
    
    # Add value labels on bars
    for bar, val in zip(bars, vals):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_multilabel_confusion_matrices(y_true, y_pred, class_names):
    """Plot individual confusion matrices for each class in multilabel classification"""
    from sklearn.metrics import multilabel_confusion_matrix
    
    print("Creating confusion matrices for each class...")
    
    # Compute multilabel confusion matrix
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    print(f"Multilabel confusion matrix shape: {mcm.shape}")
    
    # Debug: Print some statistics
    for i, (cm, class_name) in enumerate(zip(mcm, class_names)):
        print(f"Class {class_name}: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    
    # Determine grid size
    n_classes = len(class_names)
    ncols = min(3, n_classes)  # Maximum 3 columns
    nrows = (n_classes + ncols - 1) // ncols
    
    # Create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))
    if n_classes == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]
    
    for i, (cm, class_name) in enumerate(zip(mcm, class_names)):
        ax = axes[i]
        
        # Create DataFrame for better control
        df_cm = pd.DataFrame(cm, index=['Negative', 'Positive'], columns=['Negative', 'Positive'])
        
        # Plot heatmap WITHOUT mask - show all values including zeros
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', 
                   cbar=False, ax=ax, linewidths=1.0,
                   annot_kws={'size': 14, 'weight': 'bold'},
                   vmin=0)  # Ensure 0 values are properly colored
        
        ax.set_title(f'Class: {class_name}', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        
        # Rotate labels for better readability
        ax.tick_params(axis='x', labelrotation=45)
        ax.tick_params(axis='y', labelrotation=0)
    
    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Confusion Matrix for Each Class (Multilabel)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def plot_overall_confusion_matrix(y_true, y_pred, class_names):
    """Plot overall confusion matrix by treating multilabel as multiclass problem"""
    from sklearn.metrics import confusion_matrix
    
    print("Creating overall confusion matrix...")
    
    try:
        # Convert multilabel to single label by taking argmax
        y_true_single = np.argmax(y_true, axis=1)
        y_pred_single = np.argmax(y_pred, axis=1)
        
        print(f"Converting to single-label:")
        print(f"Y_true_single shape: {y_true_single.shape}, unique values: {np.unique(y_true_single)}")
        print(f"Y_pred_single shape: {y_pred_single.shape}, unique values: {np.unique(y_pred_single)}")
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true_single, y_pred_single)
        print(f"Confusion matrix shape: {cm.shape}")
        print(f"Confusion matrix:\n{cm}")
        
        # Create DataFrame for better visualization
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Create heatmap WITHOUT masking zero values
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', 
                   linewidths=1.0, square=True,
                   cbar_kws={'shrink': 0.8, 'label': 'Count'},
                   annot_kws={'size': 12, 'weight': 'bold'},
                   vmin=0)  # Show all values including zeros
        
        # Customize the plot
        plt.title('Overall Confusion Matrix (Argmax Predictions)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error creating overall confusion matrix: {e}")
        import traceback
        traceback.print_exc()
        
        # Alternative: Show label distribution
        plot_label_distribution(y_true, y_pred, class_names)


def plot_label_distribution(y_true, y_pred, class_names):
    """Plot label distribution comparison as alternative to confusion matrix"""
    print("Creating label distribution plot...")
    
    # Calculate label counts
    true_counts = y_true.sum(axis=0)
    pred_counts = y_pred.sum(axis=0)
    
    print(f"True label distribution: {dict(zip(class_names, true_counts))}")
    print(f"Predicted label distribution: {dict(zip(class_names, pred_counts))}")
    
    # Create comparison plot
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width/2, true_counts, width, label='True Labels', 
                   color='#4e79a7', alpha=0.8, edgecolor='black')
    bars2 = plt.bar(x + width/2, pred_counts, width, label='Predicted Labels', 
                   color='#f28e2b', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Label Distribution: True vs Predicted', fontsize=14, fontweight='bold')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def parse_space_separated_labels(label_string):
    """Parse space-separated labels as created by data splitter"""
    if pd.isna(label_string) or label_string == "" or label_string is None:
        return []
    
    # Split by space and filter empty strings
    labels = [label.strip() for label in str(label_string).split(" ") if label.strip()]
    return labels


def main():
    parser = argparse.ArgumentParser(description="Predict labels from trained artifacts.")
    parser.add_argument("--model", type=str, choices=MODELS.keys(), default=DEFAULT_MODEL_NAME)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K_FALLBACK)
    parser.add_argument("--texts_file", type=str, default="val_split.csv")
    args = parser.parse_args()

    print("="*60)
    print("STARTING PREDICTION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    print(f"Top-K fallback: {args.top_k}")
    print(f"Data file: {args.texts_file}")

    # Setup paths
    art_dir, csv_out_dir = setup_paths()
    test_path = Path(csv_out_dir) / args.texts_file
    model_path = Path(art_dir) / f"{args.model}.joblib"
    mlb_path = Path(art_dir) / "label_binarizer.joblib"
    config_path = Path(art_dir) / "split_config.json"

    print(f"\nPaths:")
    print(f"- Data: {test_path}")
    print(f"- Model: {model_path}")
    print(f"- Label binarizer: {mlb_path}")

    # Check file existence
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    if not test_path.exists():
        print(f"Error: Data file not found at {test_path}")
        return

    # Load configuration if available
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"\nConfiguration from split_config:")
            print(f"- Categories: {config.get('categories', [])}")
            print(f"- Classes: {config.get('classes', [])}")
            print(f"- Total samples: {config.get('total_samples', 'Unknown')}")
        except Exception as e:
            print(f"Cannot read config: {e}")
    
    # Load model
    print(f"\nLoading model...")
    arts = load(model_path)
    if hasattr(arts, "pipeline"):
        pipe = arts.pipeline
        mlb = getattr(arts, "mlb", None)
        print("Loaded pipeline model")
    else:
        pipe = arts
        mlb = None
        print("Loaded direct model")

    # Load label binarizer
    if mlb is None and mlb_path.exists():
        try:
            mlb = load(mlb_path)
            print(f"Loaded MultiLabelBinarizer with {len(mlb.classes_)} classes")
            print(f"Classes: {list(mlb.classes_)}")
        except Exception as e:
            print(f"Cannot load MultiLabelBinarizer: {e}")
            return
    elif mlb is not None:
        print(f"Using MLB from model with {len(mlb.classes_)} classes")

    if mlb is None:
        print("No MultiLabelBinarizer available - cannot continue")
        return

    # Load data
    print(f"\nReading data from {test_path}")
    try:
        df = pd.read_csv(test_path, encoding="utf-8")
        print(f"Read {len(df)} rows of data")
        print(f"Columns: {list(df.columns)}")
        
        if "text_clean" not in df.columns:
            raise KeyError("Missing 'text_clean' column in CSV file")
        
        texts = df["text_clean"].astype(str).fillna("").tolist()
        print(f"Read {len(texts)} texts")
        
        # Read labels correctly - space-separated as created by data splitter
        labels = None
        if "label" in df.columns:
            label_strings = df["label"].fillna("").astype(str).tolist()
            print(f"Read {len(label_strings)} label strings")
            
            # Show some example labels
            print("Example label strings:")
            for i, lbl_str in enumerate(label_strings[:3]):
                parsed = parse_space_separated_labels(lbl_str)
                print(f"   {i+1}: '{lbl_str}' -> {parsed}")
            
            labels = label_strings
        else:
            print("No 'label' column found - performing prediction only")
            
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # Make predictions
    print(f"\nSTARTING PREDICTION")
    print("="*60)
    try:
        y_pred = _predict_labels(pipe, texts, mlb, args.threshold, args.top_k)
        print(f"Completed prediction with shape: {y_pred.shape}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return

    # Convert predictions to labels
    print(f"\nConverting predictions to labels...")
    try:
        labels_out = [list(label_tuple) for label_tuple in mlb.inverse_transform(y_pred)]
        print(f"Converted {len(labels_out)} predictions")
    except Exception as e:
        print(f"Error converting labels: {e}")
        labels_out = [["unknown"] for _ in range(len(y_pred))]

    # Display predictions
    print(f"\nPREDICTION RESULTS")
    print("="*60)
    
    for i, (text, labs) in enumerate(zip(texts[:5], labels_out[:5])): 
        print(f"\nSample {i+1}:")
        print(f"   Text: {text[:150]}{'...' if len(text) > 150 else ''}")
        print(f"   Prediction: {labs}")

    if len(texts) > 5:
        print(f"\n... and {len(texts) - 5} more samples")

    # Evaluate if true labels available
    if labels is not None:
        print(f"\nEVALUATION RESULTS")
        print("="*60)
        
        try:
            # Parse true labels using the same format as data splitter
            y_true_labels = [parse_space_separated_labels(s) for s in labels]
            print(f"Parsed {len(y_true_labels)} true label lists")
            
            # Show some examples
            print("Example parsed true labels:")
            for i in range(min(3, len(y_true_labels))):
                print(f"   {i+1}: {y_true_labels[i]}")
            
            # Transform to binary matrix
            Y_true = mlb.transform(y_true_labels)
            Y_pred = y_pred
            
            print(f"\nMatrix dimensions:")
            print(f"   Y_true: {Y_true.shape}")
            print(f"   Y_pred: {Y_pred.shape}")
            
            # Ensure compatible shapes
            if Y_true.shape != Y_pred.shape:
                print(f"Shape mismatch - trimming to common size")
                min_rows = min(Y_true.shape[0], Y_pred.shape[0])
                min_cols = min(Y_true.shape[1], Y_pred.shape[1])
                Y_true = Y_true[:min_rows, :min_cols]
                Y_pred = Y_pred[:min_rows, :min_cols]
                print(f"   After trimming: Y_true {Y_true.shape}, Y_pred {Y_pred.shape}")

            # Compute metrics
            scores = compute_metrics(Y_true, Y_pred, average="macro")
            
            print(f"\nFINAL RESULTS:")
            print("="*40)
            print(f"Accuracy:  {scores['accuracy']:.4f}")
            print(f"Precision: {scores['precision']:.4f}")
            print(f"Recall:    {scores['recall']:.4f}")
            print(f"F1-score:  {scores['f1']:.4f}")
            print(f"Parameters: threshold={args.threshold:.2f}, top_k={args.top_k}")

            # Plot metrics
            print(f"\nPlotting results...")
            plot_single_metric(scores["accuracy"], "Accuracy")
            plot_single_metric(scores["precision"], "Precision")
            plot_single_metric(scores["recall"], "Recall")
            plot_single_metric(scores["f1"], "F1-Score")
            plot_overview(scores, f"Results Overview - Model: {args.model.upper()}")
            
            # Plot confusion matrices with detailed debugging
            print(f"\nPlotting confusion matrices...")
            plot_multilabel_confusion_matrices(Y_true, Y_pred, list(mlb.classes_))
            plot_overall_confusion_matrix(Y_true, Y_pred, list(mlb.classes_))
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nCOMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()
