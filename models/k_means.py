from __future__ import annotations
import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from .base.base import BaseModel

class _KMeansMajority(ClassifierMixin, BaseEstimator):
    def __init__(self, n_clusters=5, random_state=42, max_iter=300):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.kmeans_ = None
        self.label_map_ = None
        self.n_labels_ = None

    def fit(self, X, Y=None):
        if Y is None:
            raise ValueError("Need Y to learn cluster→label mapping.")
        X = np.asarray(X); Y = np.asarray(Y)
        self.n_labels_ = Y.shape[1]
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init="auto",
                              random_state=self.random_state, max_iter=self.max_iter).fit(X)
        clusters = self.kmeans_.labels_
        self.label_map_ = np.zeros(self.n_clusters, dtype=int)
        for c in range(self.n_clusters):
            idx = (clusters == c)
            self.label_map_[c] = 0 if not idx.any() else int(np.argmax(Y[idx].sum(axis=0)))
        return self

    def predict(self, X):
        if self.kmeans_ is None or self.label_map_ is None or self.n_labels_ is None:
            raise RuntimeError("Model not fitted.")
        cids = self.kmeans_.predict(np.asarray(X))
        Yp = np.zeros((len(cids), self.n_labels_), dtype=int)
        Yp[np.arange(len(cids)), self.label_map_[cids]] = 1
        return Yp

class KMeansModel(BaseModel):
    def __init__(self, n_clusters: int = 5, random_state: int = 42, max_iter: int = 300, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter

    def _build_estimator(self):
        return _KMeansMajority(self.n_clusters, self.random_state, self.max_iter)

    def _wrap_supervised(self, estimator):
        # keep unsupervised: do not wrap in OvR
        return estimator
