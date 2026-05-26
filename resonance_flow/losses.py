import jax.numpy as jnp
from jax_md import space


def get_steric_clash_loss(box_size=None):
    """
    Returns a function to compute the steric clash (atom overlap) penalty.

    Args:
        box_size: Optional. If provided, uses periodic boundary conditions.
                  Otherwise, assumes free space.
    """
    if box_size is None:
        displacement_fn, _ = space.free()
    else:
        displacement_fn, _ = space.periodic(box_size)

    space_metric = space.metric(displacement_fn)

    def steric_clash_loss(positions, atom_radii):
        """
        Computes the penalty for overlapping atoms.

        Args:
            positions: (N, 3) array of atomic coordinates.
            atom_radii: (N,) array of atomic van der Waals radii.

        Returns:
            A scalar loss representing the total steric clash penalty.
        """
        dr = space.map_product(space_metric)(positions, positions)
        radii_sum = atom_radii[:, None] + atom_radii[None, :]
        overlap = jnp.maximum(radii_sum - dr, 0.0)

        mask = 1.0 - jnp.eye(positions.shape[0])
        overlap = overlap * mask

        loss = jnp.sum(overlap**2) / 2.0
        return loss

    return steric_clash_loss


def get_bond_length_loss(target_distance=1.52):
    """
    Penalizes deviations from ideal bond lengths (e.g., C-alpha to C-alpha).

    Args:
        target_distance: Ideal distance between consecutive atoms in Angstroms.
    """

    def bond_length_loss(positions):
        # Compute distances between consecutive atoms
        diffs = positions[1:] - positions[:-1]
        distances = jnp.linalg.norm(diffs, axis=-1)
        return jnp.mean((distances - target_distance) ** 2)

    return bond_length_loss


def rdc_loss(predicted_vectors, measured_rdcs, d_max=21700.0):
    """
    Scientifically correct RDC loss using Saupe tensor fitting.
    Fits the alignment tensor to the structure, then calculates the residual.

    Args:
        predicted_vectors: (N, 3) internuclear vectors from the model.
        measured_rdcs: (N,) experimental RDC values.
        d_max: Maximum dipolar coupling constant (e.g., ~21.7 kHz for N-H).

    Returns:
        Scalar loss (Mean Squared Error).
    """
    # 1. Normalize vectors
    norms = jnp.linalg.norm(predicted_vectors, axis=-1, keepdims=True)
    v = predicted_vectors / (norms + 1e-8)

    # 2. Fit Saupe Tensor (Least Squares)
    # D = d_max * [ Sxx(x^2 - z^2) + Syy(y^2 - z^2) + 2Sxy*xy + 2Sxz*xz + 2Syz*yz ]
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    A = d_max * jnp.stack(
        [x**2 - z**2, y**2 - z**2, 2 * x * y, 2 * x * z, 2 * y * z], axis=1
    )

    # Solve A * s = measured_rdcs
    # Using a small ridge penalty for numerical stability
    s, _, _, _ = jnp.linalg.lstsq(A, measured_rdcs, rcond=1e-5)

    # 3. Calculate predicted RDCs using the fitted tensor
    predicted_rdcs = A @ s

    # 4. Return MSE
    return jnp.mean((predicted_rdcs - measured_rdcs) ** 2)
