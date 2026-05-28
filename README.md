# 🧬 ResonanceFlow: Differentiable Protein Structure Prediction

[![PyPI version](https://img.shields.io/pypi/v/resonance-flow.svg)](https://pypi.org/project/resonance-flow/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/resonance-flow.svg)](https://pypi.org/project/resonance-flow/)
[![Tests](https://github.com/elkins/resonance-flow/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/resonance-flow/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/Accelerated_by-JAX-blue.svg)](https://github.com/google/jax)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

ResonanceFlow is an end-to-end differentiable framework for protein structure prediction that incorporates NMR experimental constraints directly into the folding process.

---

### 🧪 For Structural Biologists
*   **Experimental Self-Correction:** Instead of just predicting a structure, ResonanceFlow uses NMR observables (RDCs, NOEs, Chemical Shifts) to "steer" the model toward the experimental reality.
*   **Physics-Grounded:** Built on the same biophysical kernels used in standard NMR suites, but optimized for the AI era.

### 🤖 For Machine Learning Geeks
*   **Differentiable Physics Loss:** The entire NMR back-calculation is implemented as a differentiable JAX operator, allowing gradients to flow from experimental residuals back to Transformer weights.
*   **Structural Refinement:** Uses a coordination-space predictor that can be fine-tuned on a single protein using only its NMR spectrum as supervision.

---

## 🚀 Key Features

*   **Differentiable NMR Kernels:** Back-calculate RDCs, NOEs, and Chemical Shifts with full gradient support.
*   **Transformer-Based Folding:** Predicts 3D coordinates directly from amino acid sequences.
*   **Self-Correction Loop:** Minimizes the residual between back-calculated and experimental spectra during inference.

## 📦 Installation

```bash
pip install resonance-flow
```

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
