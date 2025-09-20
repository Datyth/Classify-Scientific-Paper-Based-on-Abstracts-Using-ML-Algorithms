# =============================================
# scripts/evaluate.py  (evaluate a saved model on a CSV)
# =============================================
import argparse, joblib, re
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss


def parse_labels(s):
    if pd.isna(s): return []
    s = str(s).replace(";", " ").replace(",", " ")
    return [t for t in s.split() if t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    artifacts = joblib.load(args.model)
    pipe = artifacts.pipeline
    mlb = artifacts.mlb

    df = pd.read_csv(args.csv)
    texts = df["text_clean"].astype(str).tolist()

    if mlb is None or "label" not in df.columns:
        print("Model has no labels or CSV lacks labels; evaluation skipped.")
        return

    y_list = df["label"].apply(parse_labels).tolist()
    Y_true = mlb.transform(y_list)
    Y_pred = pipe.predict(texts)

    print("Micro F1:", f1_score(Y_true, Y_pred, average="micro", zero_division=0))
    print("Macro F1:", f1_score(Y_true, Y_pred, average="macro", zero_division=0))
    print("Hamming loss:", hamming_loss(Y_true, Y_pred))

if __name__ == "__main__":
    main()
