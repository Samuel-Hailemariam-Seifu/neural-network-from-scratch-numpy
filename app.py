"""
Interactive UI for training a tiny neural network on XOR.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Ensure local "src" package imports work when app runs from repository root.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from nn import BinaryCrossEntropy, Dense, ReLU, SGD, Sequential, Sigmoid


def build_xor_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return canonical XOR dataset."""
    x_data = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y_data = np.array([[0.0], [1.0], [1.0], [0.0]])
    return x_data, y_data


def build_model(hidden_units: int, learning_rate: float, seed: int) -> Sequential:
    """
    Create a simple MLP for XOR:
    2 inputs -> hidden layer -> 1 sigmoid output.
    """
    model = Sequential(
        layers=[
            Dense(2, hidden_units, seed=seed),
            ReLU(),
            Dense(hidden_units, 1, seed=seed + 1),
            Sigmoid(),
        ]
    )
    model.compile(loss_fn=BinaryCrossEntropy(), optimizer=SGD(lr=learning_rate))
    return model


def main() -> None:
    st.set_page_config(page_title="Neural Network from Scratch", page_icon="NN", layout="wide")
    st.title("Neural Network from Scratch (NumPy)")
    st.caption("Interactive XOR trainer built with your custom NumPy network classes.")

    st.markdown(
        """
This UI lets you experiment with core training settings and immediately see:
- how the loss curve changes
- the final predicted probabilities
- the final predicted classes vs ground truth
"""
    )

    x_train, y_train = build_xor_dataset()

    with st.sidebar:
        st.header("Training Controls")
        epochs = st.slider("Epochs", min_value=500, max_value=10000, value=4000, step=500)
        learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
        hidden_units = st.slider("Hidden Units", min_value=2, max_value=32, value=8, step=1)
        seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1)
        print_every = st.slider("Console Print Frequency", min_value=100, max_value=2000, value=500, step=100)

    train_clicked = st.button("Train Model", type="primary")

    if train_clicked:
        model = build_model(hidden_units=hidden_units, learning_rate=learning_rate, seed=int(seed))

        with st.spinner("Training model..."):
            # verbose=False keeps terminal clean while preserving full training history.
            history = model.fit(
                x=x_train,
                y=y_train,
                epochs=epochs,
                print_every=print_every,
                verbose=False,
            )

        probs = model.predict(x_train)
        preds = (probs >= 0.5).astype(int)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Loss Curve")
            st.line_chart(history)

        with col2:
            st.subheader("Final Loss")
            st.metric(label="Binary Cross-Entropy", value=f"{history[-1]:.6f}")

        st.subheader("Predictions")
        result_df = pd.DataFrame(
            {
                "x1": x_train[:, 0],
                "x2": x_train[:, 1],
                "pred_prob": probs[:, 0],
                "pred_class": preds[:, 0],
                "true_class": y_train.astype(int)[:, 0],
            }
        )
        st.dataframe(result_df, use_container_width=True)

        # Quick correctness signal: XOR is solved when all classes match.
        is_correct = bool(np.array_equal(preds, y_train.astype(int)))
        if is_correct:
            st.success("Model solved XOR perfectly for this run.")
        else:
            st.warning("Model has not fully solved XOR yet. Try more epochs or different settings.")
    else:
        st.info("Set your parameters in the sidebar and click 'Train Model'.")


if __name__ == "__main__":
    main()
