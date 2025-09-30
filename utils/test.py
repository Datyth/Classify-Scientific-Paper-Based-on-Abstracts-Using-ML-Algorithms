# utils/data_splitter.py

import os, sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    
def setup_paths(args):
    """Setup file paths based on arguments"""
    csv_path = os.path.join(PROJECT_ROOT, "clean_data", "data", args)
    art_dir = os.path.join(PROJECT_ROOT, "Apply-Machine-Learning-to-Classify-Scientific-Paper-Based-on-Abstracts-", "data", "artifacts")
    csv_out_dir = os.path.join(PROJECT_ROOT, "clean_data", "data", "csv")

    Path(art_dir).mkdir(parents=True, exist_ok=True)
    Path(csv_out_dir).mkdir(parents=True, exist_ok=True)

    return csv_path, art_dir, csv_out_dir


if __name__ == "__main__":
    print("Current Path = " + PROJECT_ROOT + "\n")
    test_csv_path, _, _ = setup_paths("test.csv")

    print("Test data path: " + test_csv_path)