from __future__ import annotations

from typing import Any

import numpy as np


class Sequential:
    """
    Minimal sequential model.

    Layers are executed in order for forward pass, then reverse order for backward.
    """

    def __init__(self, layers: list[object]) -> None:
        self.layers = layers
        self.loss_fn: Any = None
        self.optimizer: Any = None

    def compile(self, loss_fn: Any, optimizer: Any) -> None:
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        print_every: int = 100,
        verbose: bool = True,
    ) -> list[float]:
        if self.loss_fn is None or self.optimizer is None:
            raise RuntimeError("Call compile(loss_fn, optimizer) before fit().")

        history: list[float] = []
        for epoch in range(1, epochs + 1):
            # 1) Forward pass
            predictions = self.forward(x)

            # 2) Compute scalar loss
            loss = self.loss_fn.forward(predictions, y)
            history.append(loss)

            # 3) Backward pass from dL/dy_pred
            grad_loss = self.loss_fn.backward()
            self.backward(grad_loss)

            # 4) Update trainable parameters
            self.optimizer.step(self.layers)

            if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == epochs):
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss:.6f}")

        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
