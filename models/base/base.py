# models/base/base.py
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
from abc import ABC, abstractmethod
import joblib

from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import BaseEstimator

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

@dataclass
class FitArtifacts:
    pipeline: Pipeline
    mlb: Optional[MultiLabelBinarizer] = None  # only for supervised multi-label

class SBERTVectorizer:
    """
    Transformer that maps texts to SBERT embeddings and L2-normalizes them.
    Normalization makes Euclidean ~ cosine distance for KNN/linear models.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self._model = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        emb = self._model.encode(list(X), batch_size=64, show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=False)
        return normalize(emb) # L2
    def __getstate__(self):
        return {"model_name": self.model_name, "_model": None}

    def __setstate__(self, state):
        self.model_name = state["model_name"]
        self._model = None
class BaseModel(ABC):
    """Template for models: fit/predict/save/load using SBERT embeddings."""
    def __init__(self, **kwargs):
        self.artifacts: Optional[FitArtifacts] = None

    @abstractmethod
    def _build_estimator(self) -> BaseEstimator:
        """Return the core sklearn estimator (e.g., KNN, DecisionTree, etc.)."""

    def _wrap_supervised(self, estimator: BaseEstimator) -> BaseEstimator:
        """By default, wrap with OneVsRest for multi-label classification."""
        return OneVsRestClassifier(estimator)

    def fit(self, texts: Iterable[str], labels: Optional[Iterable[Iterable[str]]] = None) -> "BaseModel":
        """
        Fit the model on a collection of texts and optional multi‑label targets.

        A sentence‑BERT vectorizer generates embeddings for each text.  The
        resulting embeddings are fed into the estimator returned by
        ``_build_estimator``.  If labels are provided, they are binarized
        using :class:`~sklearn.preprocessing.MultiLabelBinarizer` and the
        classifier is wrapped in ``OneVsRestClassifier`` for multi‑label
        classification.  Without labels, the pipeline trains in an
        unsupervised manner (e.g. KMeans).
        """
        # Instantiate the SBERT vectorizer once.  Using a descriptive name
        # helps clarify what this component does in the pipeline.
        sbert_vectorizer = SBERTVectorizer()  # default 'all-MiniLM-L6-v2'
        steps = [("sbert", sbert_vectorizer)]

        # Build the core estimator.  In supervised settings this will be
        # wrapped by ``_wrap_supervised`` to handle multi‑label problems.
        estimator = self._build_estimator()
        classifier = self._wrap_supervised(estimator)
        steps.append(("clf", classifier))

        # Assemble the scikit‑learn pipeline.
        model_pipeline = Pipeline(steps)

        # Prepare a MultiLabelBinarizer if labels are given.  Use a clear
        # variable name so downstream code is explicit.
        label_binarizer: Optional[MultiLabelBinarizer] = None
        if labels is not None:
            # Convert labels into a list of lists of strings.  Accept either
            # nested iterables or semi‑colon/space/comma separated strings.
            label_sequences: List[List[str]] = []
            for y in labels:
                if y is None:
                    label_sequences.append([])
                elif isinstance(y, str):
                    # Split on semicolons or commas and drop empty tokens
                    tokens = [t for t in str(y).replace(";", " ").replace(",", " ").split() if t]
                    label_sequences.append(tokens)
                else:
                    label_sequences.append([str(t).strip() for t in y if str(t).strip()])
            label_binarizer = MultiLabelBinarizer()
            label_matrix = label_binarizer.fit_transform(label_sequences)
            # Fit the pipeline on texts and their binary label matrix.  The
            # ``list(texts)`` call forces evaluation of the generator if needed.
            model_pipeline.fit(list(texts), label_matrix)
        else:
            # Unsupervised or single‑label case: fit only on texts.
            model_pipeline.fit(list(texts))

        # Save fitted pipeline and label binarizer in FitArtifacts.  Use
        # descriptive attribute names for clarity.
        self.artifacts = FitArtifacts(pipeline=model_pipeline, mlb=label_binarizer)
        return self

    def predict(self, texts: Iterable[str]) -> List[List[str]]:
        if not self.artifacts:
            raise RuntimeError("Model not fitted.")
        preds = self.artifacts.pipeline.predict(list(texts))
        if self.artifacts.mlb is not None:
            return [list(lbls) for lbls in self.artifacts.mlb.inverse_transform(preds)]
        if hasattr(preds, "tolist"):
            return [[str(p)] for p in preds.tolist()]
        return [[str(p)] for p in preds]

    def save(self, path: str | Path) -> Path:
        if not self.artifacts:
            raise RuntimeError("Nothing to save. Fit the model first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.artifacts, path, compress=3, protocol=5)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        obj = cls.__new__(cls)  # type: ignore
        obj.__init__()          # default init
        obj.artifacts = joblib.load(path)
        return obj