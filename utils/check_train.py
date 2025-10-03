#check_train_counts.py
import pandas as pd, re
from collections import Counter
from pathlib import Path

ROOT = Path(r"D:\College\_hk5\AItesting\Apply-Machine-Learning-to-Classify-Scientific-Paper-Based-on-Abstracts-")
train_csv = ROOT / "clean_data" / "splitted_data" / "train_split.csv"

wanted = ['stat', 'cond-mat', 'cs', 'math', 'physics']

def parse_labels(s):
    if pd.isna(s): return []
    toks = re.split(r'[,\;/\|]|\s+', str(s).strip())
    return [t for t in toks if t]

def top(lbl): 
    return lbl.split('.', 1)[0]

df_tr = pd.read_csv(train_csv)
cnt = Counter()
for s in df_tr["label"].dropna():
    cnt.update({top(x) for x in parse_labels(s)})

print("Train top-level counts:")
for k in wanted:
    print(f"{k:10s} {cnt.get(k, 0)}")
