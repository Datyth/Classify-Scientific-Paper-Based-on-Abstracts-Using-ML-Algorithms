#models/neural_net.py
from sklearn.neural_network import MLPClassifier as _MLP
from .base.base import BaseModel

class NeuralNetClassifier(BaseModel):
    def __init__(self, hidden_layer_sizes=(512,), activation="relu", alpha=1e-4,
                 batch_size=128, learning_rate_init=1e-3, max_iter=50, **kwargs):
        super().__init__(**kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter

    def _build_estimator(self):
        return _MLP(hidden_layer_sizes=self.hidden_layer_sizes,
                    activation=self.activation,
                    alpha=self.alpha,
                    batch_size=self.batch_size,
                    learning_rate_init=self.learning_rate_init,
                    max_iter=self.max_iter,
                    early_stopping=True,
                    n_iter_no_change=5,
                    verbose=False)
