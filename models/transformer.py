#models/transformer.py
from typing import Iterable, List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline

from .base.base import BaseModel, FitArtifacts, SBERTVectorizer  

class TransformerClassifier(BaseModel):
    """Classifier using SBERT embeddings."""
    def __init__(self, *, model_name: str = 'all-MiniLM-L6-v2', **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self._sbert_vectorizer = SBERTVectorizer(model_name=self.model_name)

    def fit(self, texts: Iterable[str], labels: Optional[Iterable[List[str]]] = None):
        # Prepare labels
        mlb = None
        Y = None
        if labels is not None:
            mlb = MultiLabelBinarizer()
            Y = mlb.fit_transform(labels)

        estimator = LogisticRegression(max_iter=1000)
        clf = OneVsRestClassifier(estimator)

        pipe = Pipeline([
            ("sbert", self._sbert_vectorizer),
            ("clf", clf),
        ])

        if Y is not None:
            pipe.fit(list(texts), Y)
        else:
            pipe.fit(list(texts), np.zeros((len(list(texts)), 1)))

        self.artifacts = FitArtifacts(pipeline=pipe, mlb=mlb)
        return self

    def predict(self, texts: Iterable[str]) -> List[List[str]]:
        if not self.artifacts or self.artifacts.pipeline is None:
            raise RuntimeError("Model not fitted.")
        y_pred = self.artifacts.pipeline.predict(list(texts))
        if self.artifacts.mlb is not None:
            return [list(lbls) for lbls in self.artifacts.mlb.inverse_transform(y_pred)]
        return [[] for _ in texts]
