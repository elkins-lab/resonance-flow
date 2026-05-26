from resonance_flow.train import main


def test_full_training_run():
    """Verify the end-to-end training loop runs without error."""
    # Run for just 2 steps to verify integration
    state = main(num_steps=2)
    assert state is not None
    print("Full training run test passed!")
