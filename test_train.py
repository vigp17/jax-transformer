from jax import random

from train import load_params, save_params, train_model

def test_train_model_decreases_loss_and_checkpoint_round_trip(tmp_path):
    params, losses = train_model(key=random.PRNGKey(4), epochs=2)
    checkpoint_path = tmp_path / "params.pkl"

    save_params(params, checkpoint_path)
    loaded_params = load_params(checkpoint_path)

    assert len(losses) == 2
    assert losses[-1] < losses[0]
    assert checkpoint_path.exists()
    assert loaded_params.keys() == params.keys()
