# 🧬 ResonanceFlow: Differentiable Protein Folding with NMR "Self-Correction"

[![Tests](https://github.com/elkins/resonance-flow/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/resonance-flow/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://elkins.github.io/resonance-flow/)

**ResonanceFlow** is a JAX-native protein folding framework that integrates differentiable biophysics with experimental NMR constraints. It allows models to "self-correct" by propagating gradients from physical violations (like atomic clashes) and NMR observables (like RDCs) back into the neural network architecture.

## 🚀 Key Features

- **JAX-Native Gradient Flow:** End-to-end differentiability from experimental constraints to model weights.
- **Biophysical Realism:** Differentiable steric clash penalties powered by `jax-md`.
- **NMR-Guided Optimization:** Built-in support for Residual Dipolar Couplings (RDCs) as structural anchors.
- **Transformer-to-Coords:** A scalable architecture for mapping sequences directly to physical 3D space.

## 🧠 The Concept: "Self-Correction"

Traditional folding models are often trained on static PDB snapshots. ResonanceFlow allows a model to "listen" to physical laws during training:
1. **Predict:** Generate 3D coordinates.
2. **Constrain:** Evaluate physical violations (clashes, bad geometry) and NMR mismatches.
3. **Correct:** Use the exact gradient of the violation to "push" the atoms—and the model weights—towards a physically valid state.

## 🛠️ Installation

```bash
pip install resonance-flow
```

Or for development:

```bash
git clone https://github.com/elkins/resonance-flow.git
cd resonance-flow
pip install -e ".[dev]"
```

## 🧪 Quick Start

Run the prototype training loop to see the model self-correct from a random state:

```python
from resonance_flow.train import main
main()
```

## 📚 Documentation

For theory, API reference, and examples, visit the [Documentation](https://elkins.github.io/resonance-flow/).

## ⚖️ License

MIT
