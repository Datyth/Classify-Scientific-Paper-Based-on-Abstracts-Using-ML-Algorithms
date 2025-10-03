#models/base/base.py
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
    mlb: Optional[MultiLabelBinarizer] = None


class SBERTVectorizer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    def fit(self, X: Iterable[str], y: None = None) -> "SBERTVectorizer":
        # Nothing to learn from the data for the vectoriser
        return self

    def transform(self, X: Iterable[str]):
        if self._model is None:
            # Load the SentenceTransformer lazily
            self._model = SentenceTransformer(self.model_name)
        # Encode the documents; normalise to unit length
        emb = self._model.encode(
            list(X),
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return normalize(emb)

    def __getstate__(self) -> dict:
        # Drop the heavy model from the state so joblib doesn't serialise it
        return {"model_name": self.model_name, "_model": None}

    def __setstate__(self, state: dict) -> None:
        self.model_name = state.get("model_name", "all-MiniLM-L6-v2")
        self._model = None


class BaseModel(ABC):
    def __init__(
        self,
        *,
        pca_components: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.artifacts: Optional[FitArtifacts] = None
        self.pca_components: Optional[int] = pca_components
        self._extra_params = kwargs

    # ------------------------------------------------------------------
    # Abstract API for subclasses

    @abstractmethod
    def _build_estimator(self) -> BaseEstimator:
        raise NotImplementedError

    def _wrap_supervised(self, estimator: BaseEstimator) -> BaseEstimator:
        return OneVsRestClassifier(estimator)

    # ------------------------------------------------------------------
    # Public API

    def fit(self, texts: Iterable[str], labels: Optional[Iterable] = None) -> "BaseModel":
        # Build pipeline steps: SBERT -> optional PCA -> classifier
        steps = []
        sbert = SBERTVectorizer()
        steps.append(("sbert", sbert))
        if self.pca_components is not None and self.pca_components > 0:
            from sklearn.decomposition import PCA
            steps.append(("pca", PCA(n_components=self.pca_components)))
        est = self._build_estimator()
        est = self._wrap_supervised(est)
        steps.append(("clf", est))
        pipe = Pipeline(steps=steps)

        mlb: Optional[MultiLabelBinarizer] = None
        # Prepare labels
        if labels is not None:
            y_list: List[List[str]] = []
            for y in labels:
                if y is None:
                    y_list.append([])
                elif isinstance(y, str):
                    # Split on common delimiters (comma, semicolon, space)
                    toks = [t for t in y.replace(";", " ").replace(",", " ").split() if t]
                    y_list.append(toks)
                else:
                    # Assume iterable of labels
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
            # Convert binary matrix back to label strings
            return [list(lbls) for lbls in self.artifacts.mlb.inverse_transform(preds)]
        # Otherwise, wrap raw predictions as strings
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
        """Load a previously saved model."""
        obj = cls.__new__(cls)
        # Call __init__ to set up defaults (pca_components and extra params)
        obj.__init__()
        obj.artifacts = joblib.load(path)
        return obj