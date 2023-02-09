from __future__ import annotations

import numpy as np


class BinaryCrossEntropy:
    """
    Binary cross-entropy loss.

    Targets y_true are expected to be shape (batch_size, 1) with values in {0,1}.
    Predictions y_pred are probabilities in (0,1), usually from Sigmoid.
    """

    def __init__(self) -> None:
        self.y_true: np.ndarray | None = None
        self.y_pred: np.ndarray | None = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        eps = 1e-12  # Prevent log(0).
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        self.y_true = y_true
        self.y_pred = y_pred
        loss = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        return float(loss)

    def backward(self) -> np.ndarray:
        if self.y_true is None or self.y_pred is None:
            raise RuntimeError("forward() must run before backward().")
        # Gradient of BCE wrt prediction probability.
        grad = (self.y_pred - self.y_true) / (self.y_pred * (1.0 - self.y_pred))
        return grad / self.y_true.shape[0]


class CategoricalCrossEntropy:
    """
    Multi-class cross-entropy for one-hot labels.

    y_true shape: (batch_size, num_classes) one-hot encoded.
    y_pred shape: (batch_size, num_classes) probabilities from Softmax.
    """

    def __init__(self) -> None:
        self.y_true: np.ndarray | None = None
        self.y_pred: np.ndarray | None = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        self.y_true = y_true
        self.y_pred = y_pred
        sample_losses = -np.sum(y_true * np.log(y_pred), axis=1)
        return float(np.mean(sample_losses))

    def backward(self) -> np.ndarray:
        if self.y_true is None or self.y_pred is None:
            raise RuntimeError("forward() must run before backward().")
        return (self.y_pred - self.y_true) / self.y_true.shape[0]
