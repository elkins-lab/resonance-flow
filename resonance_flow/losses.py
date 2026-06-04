from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
from jax_md import space


def get_steric_clash_loss(
    box_size: float | None = None, exclude_bonded_range: int = 0
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """
    Returns a function to compute the steric clash (atom overlap) penalty.

    Args:
        box_size: Optional. If provided, uses periodic boundary conditions.
                  Otherwise, assumes free space.
        exclude_bonded_range: Exclude atom pairs whose sequential index
                              separation is <= this value.  Default 0 excludes
                              only self-interactions (original behaviour).
                              Set to 1 to also exclude directly bonded 1-2
                              neighbours, or 2 for 1-2 and 1-3 pairs
                              (standard AMBER / CHARMM convention).
    """
    if box_size is None:
        displacement_fn, _ = space.free()
    else:
        displacement_fn, _ = space.periodic(box_size)

    space_metric = space.metric(displacement_fn)

    def steric_clash_loss(positions: jax.Array, atom_radii: jax.Array) -> jax.Array:
        """
        Computes the penalty for overlapping atoms.

        Args:
            positions: (N, 3) array of atomic coordinates.
            atom_radii: (N,) array of atomic van der Waals radii.

        Returns:
            A scalar loss representing the total steric clash penalty.
        """
        n = positions.shape[0]
        dr = space.map_product(space_metric)(positions, positions)
        radii_sum = atom_radii[:, None] + atom_radii[None, :]
        overlap = jnp.maximum(radii_sum - dr, 0.0)

        # Build mask: exclude pairs with index separation <= exclude_bonded_range.
        # exclude_bonded_range=0  →  only self excluded  (original behaviour)
        # exclude_bonded_range=1  →  self + 1-2 bonded excluded
        # exclude_bonded_range=2  →  self + 1-2 + 1-3 excluded
        indices = jnp.arange(n)
        pair_sep = jnp.abs(indices[:, None] - indices[None, :])
        mask = (pair_sep > exclude_bonded_range).astype(jnp.float32)
        overlap = overlap * mask

        loss = jnp.sum(overlap**2) / 2.0
        return cast(jax.Array, loss)

    return steric_clash_loss


def get_bond_length_loss(
    target_distance: float = 3.8,
) -> Callable[[jax.Array], jax.Array]:
    """
    Penalises deviations from the ideal Cα–Cα virtual bond length.

    The canonical Cα–Cα distance in a peptide chain is 3.80 ± 0.02 Å
    (Engh & Huber, Acta Crystallogr. A, 1991).  This is the virtual bond
    between sequential alpha-carbons across the full peptide unit; it is
    NOT the C–C covalent bond length (1.52 Å).

    Args:
        target_distance: Ideal Cα–Cα virtual bond length in Angstroms.
                         Default 3.8 Å (Engh & Huber 1991).
    """

    def bond_length_loss(positions: jax.Array) -> jax.Array:
        # Compute distances between consecutive Cα atoms.
        diffs = positions[1:] - positions[:-1]
        distances = jnp.linalg.norm(diffs, axis=-1)
        return cast(jax.Array, jnp.mean((distances - target_distance) ** 2))

    return bond_length_loss


def estimate_nh_proxy_vectors(ca_coords: jax.Array) -> jax.Array:
    """
    Estimates backbone N-H proxy vectors from Cα coordinates.

    Uses the anti-parallel virtual-bond approximation: for each interior
    residue i the proxy N-H direction is taken as the unit vector from
    Cα(i+1) to Cα(i-1), which is roughly anti-parallel to the local
    backbone tangent and correlates with the amide N-H orientation in
    both α-helices and β-strands.  This is a standard Cα-only coarse-
    graining strategy for alignment tensor calculations (see Zweckstetter &
    Bax, J. Am. Chem. Soc. 2000, for the geometric relationship between
    Cα positions and alignment-frame vectors).

    Note: for full-atom models, real N–H internuclear vectors should be
    supplied directly to rdc_loss instead of using this approximation.

    Args:
        ca_coords: (N, 3) array of Cα coordinates.

    Returns:
        (N-2, 3) unit proxy vectors for residues 1 … N-2.
    """
    # Anti-parallel virtual bond: Cα(i-1) − Cα(i+1), normalised.
    raw = ca_coords[:-2] - ca_coords[2:]  # shape (N-2, 3)
    norms = jnp.linalg.norm(raw, axis=-1, keepdims=True)
    return cast(jax.Array, raw / (norms + 1e-8))


def calculate_pseudo_torsions(ca_coords: jax.Array) -> jax.Array:
    """
    Calculates pseudo-torsion angles for consecutive Cα atoms.

    A pseudo-torsion is the dihedral angle formed by four consecutive
    Cα atoms (i-1, i, i+1, i+2).  In Cα-only models, these angles are
    the primary indicator of backbone conformation (analogous to
    Ramachandran angles for full-atom models).

    Typical pseudo-torsion values:
        α-helix:  ~ +50°
        β-strand: ~ ±180°

    Args:
        ca_coords: (N, 3) array of Cα coordinates.

    Returns:
        (N-3,) array of pseudo-torsion angles in degrees, range [-180, 180].
    """
    # 1. Compute bond vectors
    b1 = ca_coords[1:-2] - ca_coords[:-3]
    b2 = ca_coords[2:-1] - ca_coords[1:-2]
    b3 = ca_coords[3:] - ca_coords[2:-1]

    # 2. Compute normal vectors to the planes
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)

    # 3. Compute the angle using the atan2(y, x) robust formula.
    # n1 and n2 are normals to the two planes formed by (b1, b2) and (b2, b3).
    # The dihedral angle is the angle between these normals.
    n1_norm = n1 / (jnp.linalg.norm(n1, axis=-1, keepdims=True) + 1e-8)
    n2_norm = n2 / (jnp.linalg.norm(n2, axis=-1, keepdims=True) + 1e-8)
    b2_unit = b2 / (jnp.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)

    # y = [n1 x n2] . b2_unit
    # x = n1 . n2
    y = jnp.sum(jnp.cross(n1_norm, n2_norm) * b2_unit, axis=-1)
    x = jnp.sum(n1_norm * n2_norm, axis=-1)

    return cast(jax.Array, jnp.arctan2(y, x) * (180.0 / jnp.pi))


def fit_saupe_tensor(
    predicted_vectors: jax.Array, measured_rdcs: jax.Array, d_max: float = 21700.0
) -> jax.Array:
    """
    Fits the Saupe alignment tensor (5 components) to vectors and RDCs.

    Args:
        predicted_vectors: (N, 3) internuclear vectors.
        measured_rdcs: (N,) experimental RDC values in Hz.
        d_max: Maximum dipolar coupling constant (Hz).

    Returns:
        (5,) array containing the independent components of the Saupe tensor
        [Sxx, Syy, Sxy, Sxz, Syz].
    """
    norms = jnp.linalg.norm(predicted_vectors, axis=-1, keepdims=True)
    v = predicted_vectors / (norms + 1e-8)

    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    A = d_max * jnp.stack([x**2 - z**2, y**2 - z**2, 2 * x * y, 2 * x * z, 2 * y * z], axis=1)
    s, _, _, _ = jnp.linalg.lstsq(A, measured_rdcs, rcond=1e-5)
    return cast(jax.Array, s)


def calculate_rdcs(
    predicted_vectors: jax.Array, saupe_tensor: jax.Array, d_max: float = 21700.0
) -> jax.Array:
    """
    Back-calculates RDCs for a set of vectors given a Saupe tensor.

    Args:
        predicted_vectors: (N, 3) internuclear vectors.
        saupe_tensor: (5,) array of tensor components.
        d_max: Maximum dipolar coupling constant (Hz).

    Returns:
        (N,) predicted RDC values in Hz.
    """
    norms = jnp.linalg.norm(predicted_vectors, axis=-1, keepdims=True)
    v = predicted_vectors / (norms + 1e-8)

    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    A = d_max * jnp.stack([x**2 - z**2, y**2 - z**2, 2 * x * y, 2 * x * z, 2 * y * z], axis=1)
    return cast(jax.Array, A @ saupe_tensor)


def rdc_loss(
    predicted_vectors: jax.Array, measured_rdcs: jax.Array, d_max: float = 21700.0
) -> jax.Array:
    """
    Scientifically correct RDC loss using Saupe tensor fitting.
    Fits the alignment tensor to the structure, then calculates the residual.

    References:
        Bax & Tjandra, J. Biomol. NMR, 1997.
        Cornilescu, Marquardt, Ottiger & Bax, J. Am. Chem. Soc., 1998.
    """
    s = fit_saupe_tensor(predicted_vectors, measured_rdcs, d_max)
    predicted_rdcs = calculate_rdcs(predicted_vectors, s, d_max)
    return jnp.mean((predicted_rdcs - measured_rdcs) ** 2)


def rdc_q_factor(
    predicted_vectors: jax.Array, measured_rdcs: jax.Array, d_max: float = 21700.0
) -> jax.Array:
    """
    Computes the RDC Q-factor (Cornilescu, Marquardt, Ottiger & Bax, JACS 1998).

    The Q-factor is the NMR analogue of the crystallographic R-factor:
        Q = RMSD(D_calc − D_obs) / RMS(D_obs)
    """
    s = fit_saupe_tensor(predicted_vectors, measured_rdcs, d_max)
    predicted_rdcs = calculate_rdcs(predicted_vectors, s, d_max)

    rmsd = jnp.sqrt(jnp.mean((predicted_rdcs - measured_rdcs) ** 2))
    rms_obs = jnp.sqrt(jnp.mean(measured_rdcs**2))
    return rmsd / (rms_obs + 1e-10)


def rdc_q_free(
    predicted_vectors: jax.Array,
    measured_rdcs: jax.Array,
    train_mask: jax.Array,
    d_max: float = 21700.0,
) -> jax.Array:
    """
    Computes the Q_free cross-validation metric (Clore & Garrett, JACS 1999).

    Fits the Saupe tensor using only data where train_mask is True, then
    calculates the Q-factor on the held-out data (where train_mask is False).
    This is the gold standard for detecting overfitting to RDCs.

    Args:
        predicted_vectors: (N, 3) internuclear vectors.
        measured_rdcs: (N,) experimental RDC values.
        train_mask: (N,) boolean mask (True = use for fitting, False = use for Q_free).
        d_max: Maximum dipolar coupling constant.

    Returns:
        Q_free (dimensionless).
    """
    # 1. Fit tensor on the training subset.
    s = fit_saupe_tensor(predicted_vectors[train_mask], measured_rdcs[train_mask], d_max)

    # 2. Evaluate on the test subset (the 'free' set).
    test_mask = ~train_mask
    v_test = predicted_vectors[test_mask]
    d_test = measured_rdcs[test_mask]

    predicted_test = calculate_rdcs(v_test, s, d_max)

    rmsd = jnp.sqrt(jnp.mean((predicted_test - d_test) ** 2))
    rms_obs = jnp.sqrt(jnp.mean(d_test**2))
    return rmsd / (rms_obs + 1e-10)


def noe_upper_bound_loss(
    positions: jax.Array, noe_pairs: jax.Array, upper_bounds: jax.Array
) -> jax.Array:
    """
    Penalises violations of NOE-derived inter-proton distance upper bounds.

    NOE distance restraints are the primary source of 3D structural
    information in protein NMR, providing upper bounds on inter-proton
    distances typically in the range 1.8–6.0 Å (Wüthrich, *NMR of Proteins
    and Nucleic Acids*, 1986; Güntert et al., J. Mol. Biol., 1997).

    A flat-bottomed harmonic penalty is applied only to upper-bound
    violations (no lower-bound penalty, since NOE cross-peaks are only
    observed when protons are close):

        L_NOE = mean( max(0, d_ij − d_upper)² )

    Args:
        positions: (N, 3) atomic coordinates in Angstroms.
        noe_pairs: (M, 2) integer array of atom-index pairs.
        upper_bounds: (M,) upper distance bounds in Angstroms.

    Returns:
        Scalar NOE violation loss.
    """
    ri = positions[noe_pairs[:, 0]]
    rj = positions[noe_pairs[:, 1]]
    dists = jnp.linalg.norm(ri - rj, axis=-1)
    violations = jnp.maximum(dists - upper_bounds, 0.0)
    return jnp.mean(violations**2)
