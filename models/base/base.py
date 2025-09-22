
from __future__ import annotations
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
        sb = SBERTVectorizer()  # default 'all-MiniLM-L6-v2'
        steps = [("sbert", sb)]

        estimator = self._build_estimator()
        clf = self._wrap_supervised(estimator)
        steps.append(("clf", clf))

        pipe = Pipeline(steps)

        mlb = None
        if labels is not None:
            # labels: iterable of iterables OR strings -> binarize
            y_list = []
            for y in labels:
                if y is None:
                    y_list.append([])
                elif isinstance(y, str):
                    toks = [t for t in str(y).replace(";", " ").replace(",", " ").split() if t]
                    y_list.append(toks)
                else:
                    y_list.append([str(t).strip() for t in y if str(t).strip()])
            mlb = MultiLabelBinarizer()
            Y = mlb.fit_transform(y_list)
            pipe.fit(list(texts), Y)
        else:
            pipe.fit(list(texts))

        self.artifacts = FitArtifacts(pipeline=pipe, mlb=mlb)
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
