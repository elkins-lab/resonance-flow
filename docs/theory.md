# Theory: Differentiable Physics in ResonanceFlow

ResonanceFlow treats protein folding as an optimization problem where the loss function is derived from physical and experimental constraints.

## Steric Clash Penalty

We use a harmonic potential to penalize overlapping atoms. Given two atoms with radii $r_i$ and $r_j$ at positions $\mathbf{x}_i$ and $\mathbf{x}_j$, the overlap is defined as:

$$o_{ij} = \max(0, (r_i + r_j) - \|\mathbf{x}_i - \mathbf{x}_j\|)$$

The total loss is:

$$\mathcal{L}_{steric} = \frac{1}{2} \sum_{i \neq j} o_{ij}^2$$

Because this function is implemented in JAX, we can compute $\nabla_{\theta} \mathcal{L}_{steric}$ where $\theta$ are the Transformer weights, allowing the model to learn to avoid collisions.

## Residual Dipolar Couplings (RDCs)

RDCs provide information about the orientation of internuclear vectors (e.g., N-H bonds) relative to an external magnetic field.

The calculated RDC for a vector $\mathbf{v}$ is:

$$D = D_{max} \cdot (\mathbf{v}^T \mathbf{S} \mathbf{v})$$

where $\mathbf{S}$ is the 3x3 traceless Saupe alignment tensor. ResonanceFlow automatically fits $\mathbf{S}$ to the current structure using a differentiable least-squares solver, then computes the gradient of the residual error.

## Bond Length Constraints

Backbone geometry is constrained using a mean-squared error loss on consecutive C$\alpha$-C$\alpha$ distances, ensuring chemical validity:

$$\mathcal{L}_{bond} = \frac{1}{N-1} \sum_{i=1}^{N-1} (\|\mathbf{x}_{i+1} - \mathbf{x}_i\| - d_{ideal})^2$$
