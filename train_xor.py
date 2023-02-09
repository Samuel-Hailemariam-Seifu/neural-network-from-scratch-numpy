"""
Train a tiny neural network on XOR using NumPy only.

Run:
    python train_xor.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make "src" importable when running this script directly.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from nn import BinaryCrossEntropy, Dense, ReLU, SGD, Sequential, Sigmoid


def main() -> None:
    # XOR inputs and labels.
    # XOR truth table:
    # (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
    x_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y_train = np.array([[0.0], [1.0], [1.0], [0.0]])

    # 2 -> 8 -> 1 architecture:
    # input(2 features) -> hidden(8 neurons, ReLU) -> output(1 neuron, Sigmoid)
    model = Sequential(
        layers=[
            Dense(2, 8, seed=42),
            ReLU(),
            Dense(8, 1, seed=7),
            Sigmoid(),
        ]
    )
    model.compile(loss_fn=BinaryCrossEntropy(), optimizer=SGD(lr=0.5))

    model.fit(x_train, y_train, epochs=5000, print_every=500)

    # Final prediction probabilities.
    probs = model.predict(x_train)
    preds = (probs >= 0.5).astype(int)

    print("\nPredicted probabilities:")
    print(np.round(probs, 4))
    print("\nPredicted classes:")
    print(preds)
    print("\nTrue classes:")
    print(y_train.astype(int))


if __name__ == "__main__":
    main()
