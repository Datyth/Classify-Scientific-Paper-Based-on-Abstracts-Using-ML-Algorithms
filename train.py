
# train.py
import os
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from joblib import dump, load
from utils.path import data_dirs
from models.base.factory import ModelFactory

MODEL_CHOICES = ModelFactory.choices()

def load_split_data(csv_out_dir: Path, art_dir: Path, text_col: str, label_col: str):
    train_path = csv_out_dir / "train_split.csv"
    train_df = pd.read_csv(train_path)
    val_path = csv_out_dir / "val_split.csv"
    val_df = pd.read_csv(val_path)
    texts_train = train_df[text_col].astype(str).tolist()
    texts_val = val_df[text_col].astype(str).tolist()


    mlb_path = art_dir / "label_binarizer.joblib"
    label_binarizer: MultiLabelBinarizer = load(mlb_path)


    def parse_labels_from_string(label_string: str) -> list[str]:
        if pd.isna(label_string) or label_string == "":
            return []
        return label_string.split()

    train_labels = train_df[label_col].apply(parse_labels_from_string).tolist()
    val_labels = val_df[label_col].apply(parse_labels_from_string).tolist()

    # Transform to binary matrices
    label_matrix_train = label_binarizer.transform(train_labels)
    label_matrix_val = label_binarizer.transform(val_labels)

    return texts_train, label_matrix_train, texts_val, label_matrix_val, label_binarizer

def setup_paths():
    art_dir, data_dir, splits_dir = data_dirs()
    return art_dir, splits_dir, data_dir

def main():
    parser = argparse.ArgumentParser(
        description="Train machine learning model for scientific paper classification")

    parser.add_argument("--model", type=str, choices=MODEL_CHOICES,
                        help=f"Name of the model to train. Choices: {MODEL_CHOICES}")

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

    # Setup paths relative to the project root
    try:
        art_dir, csv_out_dir, data_dir = setup_paths()
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

    # Load pre‑split data
    try:
        print("Loading pre‑split data...")
        texts_train, label_matrix_train, texts_val, label_matrix_val, label_binarizer = load_split_data(
            csv_out_dir, art_dir, args.text_col, args.label_col
        )
        print(f"Loaded train data: {len(texts_train)} samples")
        print(f"Loaded validation data: {len(texts_val)} samples")
    except Exception as e:
        print(f"Error loading split data: {e}")
        print("Make sure to run data_splitter.py first!")
        return 1

    # Import and create model using ModelFactory
    try:
        model = ModelFactory.create(args.model)
        print(f"Successfully created {args.model} model")
    except Exception as e:
        print(f"Failed to create {args.model} model: {e}")
        return 1

    # Train model
    try:
        print(f"Training {args.model} model...")
        # Inverse transform the binary label matrix back to lists of labels
        labels_train = label_binarizer.inverse_transform(label_matrix_train)
        model.fit(texts_train, labels=labels_train)
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
            "classes": list(label_binarizer.classes_),
            "train_samples": len(texts_train),
            "val_samples": len(texts_val),
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
