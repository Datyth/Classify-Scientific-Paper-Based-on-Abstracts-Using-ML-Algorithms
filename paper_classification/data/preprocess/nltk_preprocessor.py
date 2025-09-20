import re, string
from typing import Optional
import pandas as pd

# --- NLTK setup
import nltk
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find("tokenizers/punkt" if pkg == "punkt" else pkg)
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag

STOP = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()

URL_RE = re.compile(r'https?://\S+|www\.\S+')
NUM_RE = re.compile(r'\b\d+(\.\d+)?\b')
PUNCT_TABLE = str.maketrans('', '', string.punctuation)

__all__ = ["TextClassifier"] 

class TextClassifier:
    def __init__(self, *, use_stem: bool = False, min_token_len: int = 2):
        self.use_stem = use_stem
        self.min_token_len = min_token_len

    @staticmethod
    def _wn_pos(tag: str):
        if tag.startswith('J'): return wordnet.ADJ
        if tag.startswith('V'): return wordnet.VERB
        if tag.startswith('N'): return wordnet.NOUN
        if tag.startswith('R'): return wordnet.ADV
        return wordnet.NOUN

    # <-- make this an instance method (no @staticmethod)
    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        # 1) lowercase
        text = text.lower()
        # 2) remove urls and numbers
        text = URL_RE.sub(" ", text)
        text = NUM_RE.sub(" ", text)
        # 3) remove punctuation
        text = text.translate(PUNCT_TABLE)
        # 4) tokenize
        tokens = word_tokenize(text)
        # 5) stopwords & keep-only alphabetic
        tokens = [t for t in tokens if t.isalpha() and t not in STOP]
        # 6) lemmatize OR stem
        if self.use_stem:
            tokens = [STEMMER.stem(t) for t in tokens]
        else:
            if tokens:
                tagged = pos_tag(tokens)
                tokens = [LEMMATIZER.lemmatize(tok, self._wn_pos(pos)) for tok, pos in tagged]
        # 7) filter short tokens
        tokens = [t for t in tokens if len(t) >= self.min_token_len]
        return " ".join(tokens)

    @staticmethod
    def preprocess_hf(dataset_id: str, split="train", text_col="text", label_col: Optional[str]=None,
                      sample: Optional[int]=20000, out_csv="clean.csv", use_stem=False):
        from datasets import load_dataset
        tc = TextClassifier(use_stem=use_stem)
        ds = load_dataset(dataset_id, split=split)
        if sample and sample < len(ds):
            ds = ds.shuffle(seed=42).select(range(sample))

        def _apply(e):
            rec = {"text_clean": tc.normalize(e[text_col])}  # <-- fix here
            if label_col and label_col in e:
                val = e[label_col]
                # If label is a list (e.g., multi-category), join it:
                if isinstance(val, list):
                    val = ";".join(map(str, val))
                rec["label"] = val
            return rec

        mapped = ds.map(_apply)
        pdf = mapped.to_pandas()

        keep_cols = ["text_clean"] + (["label"] if "label" in pdf.columns else [])
        pdf[keep_cols].to_csv(out_csv, index=False)
        print(f"Wrote {len(pdf)} rows -> {out_csv}")

    @staticmethod
    def preprocess_csv(path: str, text_col="text", label_col: Optional[str]=None,
                       sample: Optional[int]=None, out_csv="clean.csv", use_stem=False):
        tc = TextClassifier(use_stem=use_stem)
        df = pd.read_csv(path)
        if sample and sample < len(df):
            df = df.sample(n=sample, random_state=42)
        out = pd.DataFrame({"text_clean": df[text_col].astype(str).apply(tc.normalize)})
        if label_col and label_col in df.columns:
            out["label"] = df[label_col]
        out.to_csv(out_csv, index=False)
        print(f"Wrote {len(out)} rows -> {out_csv}")
