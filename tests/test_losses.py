import jax
import jax.numpy as jnp

from resonance_flow.losses import get_steric_clash_loss, rdc_loss


def test_steric_clash_gradient():
    loss_fn = get_steric_clash_loss()
    positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atom_radii = jnp.array([1.0, 1.0])

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(positions, atom_radii)

    assert loss > 0.0
    assert grads[0, 0] > 0.0
    assert grads[1, 0] < 0.0


def test_rdc_gradient():
    # Use 6 vectors and random measured values to ensure they can't be perfectly fitted
    # (since the tensor only has 5 degrees of freedom).
    vectors = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    measured = jnp.array([10.0, -5.0, 2.0, 0.0, 4.0, 8.0])

    grad_fn = jax.value_and_grad(rdc_loss)
    loss, grads = grad_fn(vectors, measured)

    assert loss > 0.0
    assert grads.shape == vectors.shape
