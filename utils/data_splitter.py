# scripts/data_splitter.py
import os, sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from joblib import dump
from tqdm import tqdm

# Make package importable when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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

def _prepare_xy_topcats(df: pd.DataFrame, text_col: str, label_col: str, cats: list[str]):
    """Prepare X, Y data for top-level categories"""
    X_text = df[text_col].astype(str)
    y_raw = df[label_col].apply(_parse_labels)
    y_top = []
    keep_rows = []
    catset = set(c.lower() for c in cats)

    print(f"Processing {len(y_raw)} samples for category filtering...")

    for i, labels in enumerate(tqdm(y_raw, desc="Filtering categories")):
        mapped = list({_to_toplevel(l) for l in labels})  
        filtered = [t for t in mapped if t in catset]
        if filtered:
            y_top.append(filtered)
            keep_rows.append(True)
        else:
            keep_rows.append(False)

    # Filter rows
    X_text = X_text[keep_rows].reset_index(drop=True)
    y_top = pd.Series(y_top)
    mlb = MultiLabelBinarizer(classes=sorted(catset))
    Y = mlb.fit_transform(y_top)

    return X_text.tolist(), Y, mlb

def setup_paths(args):
    """Setup file paths based on arguments"""
    csv_path = os.path.join(PROJECT_ROOT,"clean_data", "data", args.data_file)
    art_dir = os.path.join(PROJECT_ROOT,"clean_data", "artifacts")
    csv_out_dir = os.path.join(PROJECT_ROOT, "clean_data","splitted_data")

    Path(art_dir).mkdir(parents=True, exist_ok=True)
    Path(csv_out_dir).mkdir(parents=True, exist_ok=True)
    return csv_path, art_dir, csv_out_dir

def main():
    parser = argparse.ArgumentParser(
        description="Split and prepare data for machine learning model training")

    parser.add_argument("--data_file", type=str, default="arxiv_clean_sample-5k.csv",
                        help="CSV file name in the data directory")

    parser.add_argument("--text_col", type=str, default="text_clean",
                        help="Name of the text column")

    parser.add_argument("--label_col", type=str, default="label", 
                        help="Name of the label column")

    parser.add_argument("--categories", nargs='+', 
                        default=['astro-ph', 'cond-mat', 'cs', 'math', 'physics'],
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

    # Prepare X, Y data
    try:
        X_text, Y, mlb = _prepare_xy_topcats(df, args.text_col, args.label_col, args.categories)
        print(f"Prepared data - Samples kept: {len(X_text)}")
        print(f"Classes: {list(mlb.classes_)}")
        print("Label counts:")
        label_counts = pd.Series(Y.sum(axis=0), index=mlb.classes_).astype(int)
        for class_name, count in label_counts.items():
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
            idx = np.arange(len(X_text))
            (train_idx, val_idx) = next(msss.split(idx, Y))
            print("Used MultilabelStratifiedShuffleSplit")
        except ImportError:
            print("iterative-stratification not found; using random split")
            idx = np.arange(len(X_text))
            train_idx, val_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed, shuffle=True)

        # Build splits
        X_train = [X_text[i] for i in train_idx]
        Y_train = Y[train_idx]
        X_val = [X_text[i] for i in val_idx]
        Y_val = Y[val_idx]

        print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

    except Exception as e:
        print(f"Error splitting data: {e}")
        return 1

    # Save split data
    try:
        print("Saving split data...")

        # Save train split
        train_path = Path(csv_out_dir) / "train_split.csv"
        pd.DataFrame({
            args.text_col: X_train,
            args.label_col: [" ".join(labs) for labs in mlb.inverse_transform(Y_train)]
        }).to_csv(train_path, index=False)
        print(f"Saved training split -> {train_path}")

        # Save validation split
        val_path = Path(csv_out_dir) / "val_split.csv"
        pd.DataFrame({
            args.text_col: X_val,
            args.label_col: [" ".join(labs) for labs in mlb.inverse_transform(Y_val)]
        }).to_csv(val_path, index=False)
        print(f"Saved validation split -> {val_path}")

        # Save label binarizer
        mlb_path = Path(art_dir) / "label_binarizer.joblib"
        dump(mlb, mlb_path, compress=3, protocol=5)
        print(f"Saved label binarizer -> {mlb_path}")

        # Save split configuration
        config_path = Path(art_dir) / "split_config.json"
        config = {
            "categories": args.categories,
            "text_col": args.text_col,
            "label_col": args.label_col,
            "seed": args.seed,
            "test_size": args.test_size,
            "classes": list(mlb.classes_),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "total_samples": len(X_text)
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
