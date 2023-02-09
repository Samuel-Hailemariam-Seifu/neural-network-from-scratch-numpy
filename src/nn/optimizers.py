from __future__ import annotations

from typing import Iterable

from .layers import Dense


class SGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, lr: float = 0.1) -> None:
        self.lr = lr

    def step(self, layers: Iterable[object]) -> None:
        """
        Update trainable parameters in-place.

        Only Dense layers have weights/bias in this simple project.
        """
        for layer in layers:
            if isinstance(layer, Dense):
                if layer.grad_weights is None or layer.grad_bias is None:
                    raise RuntimeError("Layer gradients are missing. Did you run backward()?")
                layer.weights -= self.lr * layer.grad_weights
                layer.bias -= self.lr * layer.grad_bias
