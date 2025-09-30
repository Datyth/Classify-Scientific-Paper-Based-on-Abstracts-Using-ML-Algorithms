# utils/paths.py
from pathlib import Path

# Folders that must exist in your project root
_REQUIRED_DIRS = ("clean_data", "models", "utils")

def project_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    # search current dir and all parents
    for base in [here.parent, *here.parents]:
        if all((base / d).exists() for d in _REQUIRED_DIRS):
            return base
    # Fallback: directory containing this file
    return here.parent

def data_dirs():
    root = project_root()
    base = root / "clean_data"
    return (base / "artifacts", base / "data", base / "splitted_data")

def model_artifact(model_name: str) -> Path:
    art, _, _ = data_dirs()
    return art / f"{model_name}.joblib"

def label_binarizer_path() -> Path:
    art, _, _ = data_dirs()
    return art / "label_binarizer.joblib"

def split_csv_paths():
    _, _, splits = data_dirs()
    return (splits / "train_split.csv", splits / "val_split.csv")
