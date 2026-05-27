from typing import cast

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from resonance_flow.losses import (
    estimate_nh_proxy_vectors,
    get_bond_length_loss,
    get_steric_clash_loss,
    rdc_loss,
)
from resonance_flow.model import TransformerCoordinatePredictor

...


def create_train_state(
    rng: jax.Array, model: TransformerCoordinatePredictor, learning_rate: float = 1e-3
) -> train_state.TrainState:
    """Creates initial `TrainState`."""
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    params = model.init(rng, dummy_input, deterministic=True)["params"]
    tx = optax.adamw(learning_rate)
    return cast(
        train_state.TrainState,
        train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx),
    )


@jax.jit
def train_step(
    state: train_state.TrainState,
    batch: jax.Array,
    dropout_rng: jax.Array,
    atom_radii: jax.Array,
    measured_rdcs: jax.Array,
) -> tuple[train_state.TrainState, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Takes a single training step."""
    steric_loss_fn = get_steric_clash_loss()
    bond_loss_fn = get_bond_length_loss()

    def loss_fn(params: optax.Params) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        coords = state.apply_fn(
            {"params": params},
            batch,
            deterministic=False,
            rngs={"dropout": dropout_rng},
        )

        c = coords[0]
        l_steric = steric_loss_fn(c, atom_radii)
        l_bond = bond_loss_fn(c)

        # Estimate backbone N-H proxy vectors from Cα positions using the
        # anti-parallel virtual-bond approximation (Zweckstetter & Bax,
        # J. Am. Chem. Soc. 2000).  A full-atom model should supply true
        # N-H internuclear vectors here instead.
        nh_proxy = estimate_nh_proxy_vectors(c)  # shape (seq_len-2, 3)
        l_rdc = rdc_loss(nh_proxy, measured_rdcs)

        total_loss = l_steric + 10.0 * l_bond + l_rdc
        return total_loss, (l_steric, l_bond, l_rdc)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (l_steric, l_bond, l_rdc)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss, l_steric, l_bond, l_rdc


def main(num_steps: int = 101, learning_rate: float = 1e-2) -> train_state.TrainState:
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    model = TransformerCoordinatePredictor(vocab_size=21, d_model=32, num_heads=2, num_layers=2)

    state = create_train_state(init_rng, model, learning_rate=learning_rate)

    batch_size = 1
    seq_len = 10
    batch = jax.random.randint(rng, (batch_size, seq_len), 0, 21)
    atom_radii = jnp.ones((seq_len,)) * 1.5
    # estimate_nh_proxy_vectors returns N-2 vectors for N Cα positions.
    measured_rdcs = jnp.ones((seq_len - 2,))

    print(f"Starting ResonanceFlow training loop ({num_steps} steps)...")
    for step in range(num_steps):
        rng, dropout_rng = jax.random.split(rng)
        state, loss, l_steric, l_bond, l_rdc = train_step(
            state, batch, dropout_rng, atom_radii, measured_rdcs
        )
        if step % 10 == 0:
            print(
                f"Step {step:3d} | Total Loss: {loss:.4f} | "
                f"Steric: {l_steric:.4f} | Bond: {l_bond:.4f} | RDC: {l_rdc:.4f}"
            )
    return state


if __name__ == "__main__":
    main()
