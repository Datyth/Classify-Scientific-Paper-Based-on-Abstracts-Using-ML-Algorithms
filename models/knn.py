#models/knn.py
from sklearn.neighbors import KNeighborsClassifier as _KNNClassifier

from .base.base import BaseModel


class KNNModel(BaseModel):
    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        weights: str = "distance",
        pca_components: int | None = None,
        **kwargs,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        super().__init__(pca_components=pca_components, **kwargs)

    def _build_estimator(self) -> _KNNClassifier:
        """Construct the KNN estimator."""
        return _KNNClassifier(n_neighbors=self.n_neighbors, weights=self.weights)