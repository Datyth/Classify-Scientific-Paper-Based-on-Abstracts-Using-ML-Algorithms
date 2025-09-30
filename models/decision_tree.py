#models/decision_tree.py
from sklearn.tree import DecisionTreeClassifier
from .base.base import BaseModel

class DecisionTreeClassifier(BaseModel):
    def __init__(self, max_depth: int | None = None, class_weight: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.max_depth = max_depth
        self.class_weight = class_weight

    def _build_estimator(self):
        return DecisionTreeClassifier(max_depth=self.max_depth, class_weight=self.class_weight)