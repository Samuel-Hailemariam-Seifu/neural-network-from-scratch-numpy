from __future__ import annotations

import numpy as np


class Dense:
    """
    Fully connected neural network layer.

    Mathematical form:
        y = x @ W + b

    Shapes:
        x: (batch_size, in_features)
        W: (in_features, out_features)
        b: (1, out_features)
        y: (batch_size, out_features)
    """

    def __init__(self, in_features: int, out_features: int, seed: int | None = None) -> None:
        # He-style scale keeps activations in a healthy range for deep nets.
        rng = np.random.default_rng(seed)
        weight_scale = np.sqrt(2.0 / in_features)
        self.weights = rng.standard_normal((in_features, out_features)) * weight_scale
        self.bias = np.zeros((1, out_features))

        # Placeholders filled during forward/backward pass.
        self.inputs: np.ndarray | None = None
        self.grad_weights: np.ndarray | None = None
        self.grad_bias: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Cache inputs and compute linear transform."""
        self.inputs = x
        return x @ self.weights + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backprop through linear layer.

        Given:
            grad_output = dL/dy
        We compute:
            dL/dW = x^T @ grad_output
            dL/db = sum(grad_output over batch)
            dL/dx = grad_output @ W^T
        """
        if self.inputs is None:
            raise RuntimeError("forward() must run before backward().")

        batch_size = self.inputs.shape[0]
        self.grad_weights = (self.inputs.T @ grad_output) / batch_size
        self.grad_bias = np.sum(grad_output, axis=0, keepdims=True) / batch_size
        grad_inputs = grad_output @ self.weights.T
        return grad_inputs
