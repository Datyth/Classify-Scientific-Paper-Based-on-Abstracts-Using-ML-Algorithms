#inspect_artifact.py
from joblib import load
from pathlib import Path

# adjust if your project root differs
ROOT = Path(r"D:\College\_hk5\AItesting\Apply-Machine-Learning-to-Classify-Scientific-Paper-Based-on-Abstracts-")
art = ROOT / "clean_data" / "artifacts" / "decision_tree.joblib"

obj = load(art)
if hasattr(obj, "mlb") and obj.mlb is not None:
    mlb = obj.mlb
elif hasattr(obj, "pipeline"):
    # sometimes mlb is stored in the pipeline's steps; try common places
    mlb = getattr(obj, "mlb", None)
else:
    mlb = None

if mlb is None:
    print("No MultiLabelBinarizer found in artifact.")
else:
    print("Artifact classes (mlb.classes_):")
    print(list(mlb.classes_))
    print("Num classes:", len(mlb.classes_))
