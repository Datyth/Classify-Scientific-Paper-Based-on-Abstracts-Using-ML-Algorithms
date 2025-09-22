from collections import Counter
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from .base.base import BaseModel, FitArtifacts, SBERTVectorizer
from typing import Optional

class KMeansModel(BaseModel):
    def __init__(self, n_clusters: int = 20, random_state: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._cluster_to_label: Optional[dict[int, str]] = None

    def _build_estimator(self):
        return KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)

    def _wrap_supervised(self, estimator):
        # No wrapping for clustering
        return estimator

    def fit(self, texts, labels=None):
        # SBERT embedding followed by KMeans clustering
        sb = SBERTVectorizer()
        steps = [("sbert", sb), ("kmeans", self._build_estimator())]
        pipe = Pipeline(steps)
        pipe.fit(list(texts))
        self.artifacts = FitArtifacts(pipeline=pipe, mlb=None)

        # 1 single label case
        if labels is not None:
            y_simple = []
            for y in labels:
                if y is None:
                    y_simple.append(None)
                elif isinstance(y, str):
                    lab = [t for t in y.replace(";", " ").replace(",", " ").split() if t]
                    y_simple.append(lab[0] if lab else None)
                else:
                    y_simple.append(str(y[0]).strip() if len(y) else None)
            clusters = pipe.predict(list(texts))
            mapping: dict[int, str] = {}
            for c in np.unique(clusters):
                idx = np.where(clusters == c)[0]
                labs = [y_simple[i] for i in idx if y_simple[i] is not None]
                mapping[int(c)] = Counter(labs).most_common(1)[0][0] if labs else str(int(c))
            self._cluster_to_label = mapping

        return self

    def predict(self, texts):
        if not self.artifacts:
            raise RuntimeError("Model not fitted.")
        clusters = self.artifacts.pipeline.predict(list(texts))
        if self._cluster_to_label:
            return [[self._cluster_to_label.get(int(c), str(int(c)))] for c in clusters]
        return [[str(int(c))] for c in clusters]
