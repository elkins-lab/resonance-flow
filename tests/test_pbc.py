import jax
import jax.numpy as jnp

from resonance_flow.losses import get_steric_clash_loss


def test_periodic_boundary_conditions():
    """Verify steric clash loss works with PBC."""
    # Atoms at opposite ends of a 10.0 Angstrom box
    box_size = 10.0
    loss_fn = get_steric_clash_loss(box_size=box_size)

    # Atom 1 at 0.1, Atom 2 at 9.9
    # Real distance = 0.2 across boundary
    positions = jnp.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]])
    atom_radii = jnp.array([1.0, 1.0])  # Radii sum = 2.0

    loss = loss_fn(positions, atom_radii)
    assert loss > 0.0, f"Expected collision across PBC boundary, got loss {loss}"

    # Check gradients push them away across the boundary
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(positions, atom_radii)

    # Atom 1 should be pushed right (positive x) away from 9.9
    # Atom 2 should be pushed left (negative x) away from 0.1
    assert grads[0, 0] < 0.0  # Moving it left increases loss (closer to 9.9)
    assert grads[1, 0] > 0.0
    print("PBC steric clash test passed!")


if __name__ == "__main__":
    test_periodic_boundary_conditions()
