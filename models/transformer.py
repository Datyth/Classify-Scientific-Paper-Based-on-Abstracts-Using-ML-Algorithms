#models/transformer.py
from sklearn.linear_model import LogisticRegression
from .base.base import BaseModel

class TransformerModel(BaseModel):
    def __init__(self, C: float = 1.0, max_iter: int = 1000, class_weight: str | None = "balanced",
                 solver: str = "liblinear", **kwargs):
        super().__init__(**kwargs)
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.solver = solver

    def _build_estimator(self):
        return LogisticRegression(
            C=self.C, max_iter=self.max_iter, class_weight=self.class_weight, solver=self.solver
        )
