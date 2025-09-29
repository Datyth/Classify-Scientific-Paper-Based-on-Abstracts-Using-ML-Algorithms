# scripts/train_model.py
import os, sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from joblib import dump, load

# Make package importable when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Available models dictionary
MODELS = {
    "knn": "models.knn.KNNClassifier",
    "decision_tree": "models.decision_tree.DecisionTreeClassifier", 
    "kmeans": "models.k_means.KMeansClassifier",
    "transformer": "models.transformer.TransformerModel"
}

def import_model_class(model_key: str):
    """
    Dynamically import model class only when needed to avoid environment conflicts.

    Args:
        model_key: Key from MODELS dict

    Returns:
        Model class
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

    module_path, class_name = MODELS[model_key].rsplit('.', 1)

    try:
        print(f"Importing {class_name} from {module_path}...")
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)
        return model_class
    except ImportError as e:
        print(f"Failed to import {class_name}: {e}")
        print(f"Make sure the required dependencies for {model_key} are installed")
        raise
    except AttributeError as e:
        print(f"Class {class_name} not found in {module_path}: {e}")
        raise

def load_split_data(csv_out_dir, art_dir, text_col, label_col):
    """Load pre-split data and label binarizer"""

    # Load train data
    train_path = Path(csv_out_dir) / "train_split.csv"
    train_df = pd.read_csv(train_path)
    X_train = train_df[text_col].tolist()

    # Load validation data  
    val_path = Path(csv_out_dir) / "val_split.csv"
    val_df = pd.read_csv(val_path)
    X_val = val_df[text_col].tolist()

    # Load label binarizer
    mlb_path = Path(art_dir) / "label_binarizer.joblib"
    mlb = load(mlb_path)

    # Convert labels back to binary format
    def parse_labels_from_string(label_string):
        if pd.isna(label_string) or label_string == "":
            return []
        return label_string.split()

    train_labels = train_df[label_col].apply(parse_labels_from_string).tolist()
    val_labels = val_df[label_col].apply(parse_labels_from_string).tolist()

    Y_train = mlb.transform(train_labels)
    Y_val = mlb.transform(val_labels)

    return X_train, Y_train, X_val, Y_val, mlb

def setup_paths():
    """Setup file paths"""
    art_dir = os.path.join(PROJECT_ROOT, "Apply-Machine-Learning-to-Classify-Scientific-Paper-Based-on-Abstracts-", "clean_data", "artifacts")
    csv_out_dir = os.path.join(PROJECT_ROOT, "Apply-Machine-Learning-to-Classify-Scientific-Paper-Based-on-Abstracts-", "clean_data", "splitted_data")
    return art_dir, csv_out_dir

def main():
    parser = argparse.ArgumentParser(
        description="Train machine learning model for scientific paper classification")

    parser.add_argument("--model", type=str, choices=MODELS.keys(),
                        help=f"Name of the model to train. Choices: {list(MODELS.keys())}")

    parser.add_argument("--text_col", type=str, default="text_clean",
                        help="Name of the text column")

    parser.add_argument("--label_col", type=str, default="label", 
                        help="Name of the label column")

    args = parser.parse_args()

    print(f"Starting training with {args.model} model...")
    print(f"Configuration:")
    print(f"   - Model: {args.model}")
    print(f"   - Text column: {args.text_col}")
    print(f"   - Label column: {args.label_col}")

    # Setup paths
    try:
        art_dir, csv_out_dir = setup_paths()
        print(f"Artifacts dir: {art_dir}")
        print(f"CSV output dir: {csv_out_dir}")
    except Exception as e:
        print(f"Error setting up paths: {e}")
        return 1

    # Load split configuration
    try:
        config_path = Path(art_dir) / "split_config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            split_config = json.load(f)
        print("Loaded split configuration:")
        print(f"   - Classes: {split_config['classes']}")
        print(f"   - Train samples: {split_config['train_samples']}")
        print(f"   - Val samples: {split_config['val_samples']}")
    except Exception as e:
        print(f"Error loading split configuration: {e}")
        print("Make sure to run data_splitter.py first!")
        return 1

    # Load pre-split data
    try:
        print("Loading pre-split data...")
        X_train, Y_train, X_val, Y_val, mlb = load_split_data(
            csv_out_dir, art_dir, args.text_col, args.label_col
        )
        print(f"Loaded train data: {len(X_train)} samples")
        print(f"Loaded validation data: {len(X_val)} samples")
    except Exception as e:
        print(f"Error loading split data: {e}")
        print("Make sure to run data_splitter.py first!")
        return 1

    # Import and create model
    try:
        ModelClass = import_model_class(args.model)
        model = ModelClass()
        print(f"Successfully created {args.model} model")
    except Exception as e:
        print(f"Failed to create {args.model} model: {e}")
        return 1

    # Train model
    try:
        print(f"Training {args.model} model...")
        labels_train = mlb.inverse_transform(Y_train)
        model.fit(X_train, labels=labels_train)
        print("Model training completed!")
    except Exception as e:
        print(f"Error during model training: {e}")
        return 1

    # Save trained model and artifacts
    try:
        print("Saving model artifacts...")
        model_path = Path(art_dir) / f"{args.model}.joblib"

        saved_path = model.save(model_path)
        print(f"Saved model -> {saved_path}")

        # Save training configuration
        train_config_path = Path(art_dir) / f"{args.model}.train_config.json"
        train_config = {
            "model_name": args.model,
            "text_col": args.text_col,
            "label_col": args.label_col,
            "classes": list(mlb.classes_),
            "train_samples": len(X_train),
            "val_samples": len(X_val)
        }

        with open(train_config_path, 'w', encoding='utf-8') as f:
            json.dump(train_config, f, ensure_ascii=False, indent=2)
        print(f"Saved training config -> {train_config_path}")

    except Exception as e:
        print(f"Error saving artifacts: {e}")
        return 1

    print("Training completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
