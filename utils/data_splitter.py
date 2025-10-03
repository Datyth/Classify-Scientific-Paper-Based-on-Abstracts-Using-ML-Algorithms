
# utils/data_splitter.py
import os
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from joblib import dump
from tqdm import tqdm

# Compute the project root using pathlib.  Avoid mutating sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _parse_labels(raw):
    """Parse raw labels into list of clean labels"""
    import re
    if raw is None:
        return []
    toks = re.split(r"[,;/|]|\s+", str(raw).strip())
    toks = [t.strip().lower() for t in toks if t and t.strip()]
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _to_toplevel(lbl: str) -> str:
    """Convert label to top-level category"""
    if "." in lbl:
        return lbl.split(".", 1)[0].lower()
    return lbl.lower()

def prepare_xy_top_categories(df: pd.DataFrame, text_col: str, label_col: str, cats: list[str]):
    text_series = df[text_col].astype(str)
    raw_labels = df[label_col].apply(_parse_labels)
    top_labels: list[list[str]] = []
    keep_mask: list[bool] = []
    catset = set(c.lower() for c in cats)

    print(f"Processing {len(raw_labels)} samples for category filtering...")

    for labels in tqdm(raw_labels, desc="Filtering categories"):
        mapped = { _to_toplevel(l) for l in labels }
        filtered = [t for t in mapped if t in catset]
        if filtered:
            top_labels.append(filtered)
            keep_mask.append(True)
        else:
            keep_mask.append(False)

    # Filter rows
    text_series = text_series[keep_mask].reset_index(drop=True)
    # Fit the label binarizer on the filtered list of label lists
    label_binarizer = MultiLabelBinarizer(classes=sorted(catset))
    label_matrix = label_binarizer.fit_transform(top_labels)

    return text_series.tolist(), label_matrix, label_binarizer

def setup_paths(args):
    """Setup file paths based on arguments"""
    # Use pathlib.Path to build platform‑independent paths
    csv_path = PROJECT_ROOT / "clean_data" / "data" / args.data_file
    art_dir = PROJECT_ROOT / "clean_data" / "artifacts"
    csv_out_dir = PROJECT_ROOT / "clean_data" / "splitted_data"
    # Ensure the output directories exist
    art_dir.mkdir(parents=True, exist_ok=True)
    csv_out_dir.mkdir(parents=True, exist_ok=True)
    return csv_path, art_dir, csv_out_dir

def main():
    parser = argparse.ArgumentParser(
        description="Split and prepare data for machine learning model training")

    parser.add_argument("--data_file", type=str, default="arxiv_clean_sample-20k.csv",
                        help="CSV file name in the data directory")

    parser.add_argument("--text_col", type=str, default="text_clean",
                        help="Name of the text column")

    parser.add_argument("--label_col", type=str, default="label", 
                        help="Name of the label column")

    parser.add_argument("--categories", nargs='+', 
                        default=["math", "cs", "cond-mat", "astro-ph", "physics"],
                        help="List of categories to select")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test size ratio for train/validation split")

    args = parser.parse_args()

    print("Starting data preparation and splitting...")
    print("Configuration:")
    print(f"   - Data file: {args.data_file}")
    print(f"   - Categories: {args.categories}")
    print(f"   - Seed: {args.seed}")
    print(f"   - Test size: {args.test_size}")

    # Setup paths
    try:
        csv_path, art_dir, csv_out_dir = setup_paths(args)
        print(f"Data path: {csv_path}")
        print(f"Artifacts dir: {art_dir}")
        print(f"output_csv: {csv_out_dir}")
    except Exception as e:
        print(f"Error setting up paths: {e}")
        return 1

    # Load data
    try:
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path).dropna(subset=[args.text_col, args.label_col]).reset_index(drop=True)
        print(f"Loaded {len(df)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    # Prepare the text list, binary label matrix and label binarizer.  The helper
    # ``prepare_xy_top_categories`` maps hierarchical labels to top‑level
    # categories and filters out rows without allowed labels.
    try:
        texts, label_matrix, label_binarizer = prepare_xy_top_categories(
            df, args.text_col, args.label_col, args.categories
        )
        print(f"Prepared data - Samples kept: {len(texts)}")
        print(f"Classes: {list(label_binarizer.classes_)}")
        print("Label counts:")
        counts = pd.Series(label_matrix.sum(axis=0), index=label_binarizer.classes_).astype(int)
        for class_name, count in counts.items():
            print(f"   - {class_name}: {count}")
    except Exception as e:
        print(f"Error preparing data: {e}")
        return 1

    # Split data
    try:
        print("Splitting data...")
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
            msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
            idx = np.arange(len(texts))
            (train_idx, val_idx) = next(msss.split(idx, label_matrix))
            print("Used MultilabelStratifiedShuffleSplit")
        except ImportError:
            print("iterative-stratification not found; using random split")
            idx = np.arange(len(texts))
            train_idx, val_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed, shuffle=True)

        # Build splits
        texts_train = [texts[i] for i in train_idx]
        labels_train_bin = label_matrix[train_idx]
        texts_val = [texts[i] for i in val_idx]
        labels_val_bin = label_matrix[val_idx]

        print(f"Train samples: {len(texts_train)}, Validation samples: {len(texts_val)}")

    except Exception as e:
        print(f"Error splitting data: {e}")
        return 1

    # Save split data
    try:
        print("Saving split data...")

        # Save train split
        train_path = Path(csv_out_dir) / "train_split.csv"
        pd.DataFrame({
            args.text_col: texts_train,
            args.label_col: [" ".join(labs) for labs in label_binarizer.inverse_transform(labels_train_bin)]
        }).to_csv(train_path, index=False)
        print(f"Saved training split -> {train_path}")

        # Save validation split
        val_path = Path(csv_out_dir) / "val_split.csv"
        pd.DataFrame({
            args.text_col: texts_val,
            args.label_col: [" ".join(labs) for labs in label_binarizer.inverse_transform(labels_val_bin)]
        }).to_csv(val_path, index=False)
        print(f"Saved validation split -> {val_path}")

        # Save label binarizer
        mlb_path = Path(art_dir) / "label_binarizer.joblib"
        dump(label_binarizer, mlb_path, compress=3, protocol=5)
        print(f"Saved label binarizer -> {mlb_path}")

        # Save split configuration
        config_path = Path(art_dir) / "split_config.json"
        config = {
            "categories": args.categories,
            "text_col": args.text_col,
            "label_col": args.label_col,
            "seed": args.seed,
            "test_size": args.test_size,
            "classes": list(label_binarizer.classes_),
            "train_samples": len(texts_train),
            "val_samples": len(texts_val),
            "total_samples": len(texts)
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"Saved split config -> {config_path}")

    except Exception as e:
        print(f"Error saving split data: {e}")
        return 1

    print("Data preparation and splitting completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())