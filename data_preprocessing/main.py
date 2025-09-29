from nltk_preprocessor import TextProcessor
from pathlib import Path
if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out_csv = here / "arxiv_clean_sample.csv" 
    TextProcessor.preprocess_hf(
        dataset_id="gfissore/arxiv-abstracts-2021",
        split="train",
        text_col="abstract",
        label_col="categories",
        sample=2000000,
        out_csv=str(out_csv),
    )
