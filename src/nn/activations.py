from __future__ import annotations

import numpy as np


class ReLU:
    """Rectified Linear Unit: max(0, x)."""

    def __init__(self) -> None:
        self.inputs: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.inputs = x
        return np.maximum(0.0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.inputs is None:
            raise RuntimeError("forward() must run before backward().")
        grad = grad_output.copy()
        grad[self.inputs <= 0.0] = 0.0
        return grad


class Sigmoid:
    """Sigmoid activation: 1 / (1 + e^-x)."""

    def __init__(self) -> None:
        self.outputs: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Clip input to avoid overflow in exp for large magnitude values.
        x_clipped = np.clip(x, -500, 500)
        self.outputs = 1.0 / (1.0 + np.exp(-x_clipped))
        return self.outputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.outputs is None:
            raise RuntimeError("forward() must run before backward().")
        return grad_output * self.outputs * (1.0 - self.outputs)


class Softmax:
    """
    Softmax activation for multi-class outputs.

    This class stores outputs and uses a vectorized Jacobian trick during
    backward to avoid explicit loops over classes.
    """

    def __init__(self) -> None:
        self.outputs: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Stability trick: subtract row-wise max before exponentiating.
        shifted = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shifted)
        self.outputs = exps / np.sum(exps, axis=1, keepdims=True)
        return self.outputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.outputs is None:
            raise RuntimeError("forward() must run before backward().")

        # For each sample:
        # dL/dz = s * (g - sum(g * s))
        # where s = softmax output, g = upstream gradient dL/ds
        dot = np.sum(grad_output * self.outputs, axis=1, keepdims=True)
        return self.outputs * (grad_output - dot)
