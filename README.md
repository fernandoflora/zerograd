# zerograd

A minimal neural network library built from scratch in Python and NumPy, focused on understanding forward propagation, backpropagation, and gradient descent without relying on high-level deep learning frameworks.

## Roadmap

- [x] Single neuron training loop (linear regression via gradient descent)
- [x] Modular core: dense layers, activation functions, and loss functions
- [x] Unit test for non-linear classification (XOR)
<<<<<<< HEAD
- [x] MNIST handwritten digit training and model weight serialization
=======
- [ ] MNIST handwritten digit training and model weight serialization
>>>>>>> d980ce0830f8685df6bd8610d84fe03e0fc103d8
- [ ] Interactive drawing demo and CPU performance benchmarks

## Benchmarks (MNIST)

Trained on 60,000 handwritten digits and evaluated on 10,000 unseen test images using pure NumPy on CPU:

| Architecture | Epochs | Batch Size | Learning Rate | Training Time | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Softmax)` | 15 | 64 | 0.1 | 144.28s | **97.65%** |

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the XOR non-linear classification test:

```bash
python3 -m tests.test_xor
```

Train and evaluate on the MNIST dataset (downloads the dataset automatically and saves weights to `models/mnist_model.npz`):

```bash
python3 -m examples.train_mnist
```