# MyAutoGrad

A lightweight, educational implementation of automatic differentiation built from scratch using only Python and NumPy. Inspired by Andrej Karpathy's micrograd lecture series.

## Overview

MyAutoGrad provides a minimal yet complete auto-grad engine that tracks computations and computes gradients via reverse-mode differentiation (backpropagation). It demonstrates the core concepts behind modern deep learning frameworks like PyTorch, allowing you to understand and visualize how gradients flow through computational graphs.

## Features

- **Value Class**: The core scalar wrapper that tracks operations and accumulates gradients
- **Neural Network Primitives**: `Neuron`, `Layer`, and `MLP` classes for building multi-layer perceptrons
- **Activation Functions**: Support for `tanh`, `exp`, and power operations
- **Computational Graph Visualization**: Export graphs to SVG using Graphviz
- **Training Loop**: Gradient descent optimization with zero-gradient reset and weight updates

## Installation

```bash
uv sync
# or
pip install -e .
```

## Quick Start

### Building a Computational Graph

```python
from value import Value

a = Value(3.0, label='a')
b = Value(2.0, label='b')
c = a * b
c.label = 'c'
c.backward()
```

### Creating a Neural Network

```python
from value import MLP

mlp = MLP(3, [4, 4, 1])
output = mlp([2.0, 3.0, -1.0])
output.backward()
```

### Training with Gradient Descent

```python
from value import MLP, Value

mlp = MLP(3, [2, 1])
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5]]
ys = [1.0, -1.0]

for epoch in range(100):
    ypred = [mlp(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    for p in mlp.parameters():
        p.grad = 0.0

    loss.backward()

    for p in mlp.parameters():
        p.data += -0.05 * p.grad
```

## Architecture

### Value Class

The `Value` class wraps scalar data and tracks:

- **`data`**: The scalar value
- **`grad`**: The accumulated gradient
- **`_prev`**: Child nodes in the computational graph
- **`_op`**: The operation that produced this value
- **`_backward`**: The gradient computation function

Supported operations: `+`, `-`, `*`, `/`, `**`, `exp`, `tanh`, and negation

### Network Primitives

```
MLP(nin, [hidden_layer_sizes])
  └── Layer(nin, nout)
        └── Neuron(nin)
              └── weights: List[Value]
              └── bias: Value
```

## Visualization

Render computational graphs to SVG:

```python
from draw import draw_dot

dot = draw_dot(loss)
dot.render('graph', format='svg', view=True)
```

## Project Structure

```
MyAutoGrad/
├── value.py        # Core auto-grad engine and network primitives
├── draw.py         # Computational graph visualization
├── extra_tools.py  # Kitty terminal image output
├── playground.py   # Usage examples and tests
└── pyproject.toml  # Project configuration
```
