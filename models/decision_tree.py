#models/decision_tree.py
from sklearn.tree import DecisionTreeClassifier as _DecisionTreeClassifier

from .base.base import BaseModel


class DecisionTreeModel(BaseModel):

    def __init__(
        self,
        *,
        max_depth: int | None = 20,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        class_weight: str | dict | list[dict] | None = "balanced",
        pca_components: int | None = None,
        **kwargs,
    ) -> None:
        # Store tree‑specific parameters
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        super().__init__(pca_components=pca_components, **kwargs)

    def _build_estimator(self) -> _DecisionTreeClassifier:
        """Construct the underlying decision tree estimator."""
        return _DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            class_weight=self.class_weight,
        )