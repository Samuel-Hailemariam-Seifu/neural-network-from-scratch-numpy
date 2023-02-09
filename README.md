# Neural Network From Scratch (NumPy)

A clean, educational, and GitHub-ready implementation of a neural network built **from scratch** using only `numpy`.

This project is designed to help you understand the full training pipeline:
- forward pass
- loss computation
- backpropagation
- parameter updates (SGD)

## Project Structure

```text
.
├── README.md
├── app.py
├── requirements.txt
├── train_xor.py
└── src
    └── nn
        ├── __init__.py
        ├── activations.py
        ├── layers.py
        ├── losses.py
        ├── model.py
        └── optimizers.py
```

## Quick Start

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the XOR training example:

```bash
python train_xor.py
```

4. Run the interactive UI:

```bash
streamlit run app.py
```

You should see the loss decrease and predictions close to:
- `[0, 1, 1, 0]`

In the UI, you can tune epochs, learning rate, hidden layer size, and seed to see how training behavior changes.

## What Each Module Does

- `layers.py`
  - Implements `Dense` (fully connected) layer.
  - Handles gradients for `weights`, `bias`, and input tensor.

- `activations.py`
  - `ReLU`, `Sigmoid`, and `Softmax` activations.
  - Includes numerically stable softmax forward pass.

- `losses.py`
  - `BinaryCrossEntropy` for binary classification.
  - `CategoricalCrossEntropy` for multi-class one-hot targets.

- `optimizers.py`
  - `SGD` optimizer updates trainable parameters.

- `model.py`
  - `Sequential` model orchestration.
  - Defines `compile`, `fit`, `predict`.

## Learning Notes

If you want to trace one full training step:
1. `model.forward(x)` runs each layer in order.
2. `loss_fn.forward(pred, y)` computes scalar loss.
3. `loss_fn.backward()` gives gradient wrt model output.
4. `model.backward(grad)` propagates gradients backward.
5. `optimizer.step(layers)` updates trainable params.

## Next Improvements (Optional)

- mini-batch training
- validation loop and metrics
- learning rate scheduling
- model save/load utilities
- gradient checking tests

## License

MIT (or your preferred open-source license).
