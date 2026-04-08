from jax import random

from train import decode, encode, load_checkpoint, save_checkpoint, train_model

def test_train_model_decreases_loss_and_checkpoint_round_trip(tmp_path):
    text = "hello transformer\n" * 20
    params, losses, metadata = train_model(
        key=random.PRNGKey(4),
        epochs=2,
        block_size=8,
        batch_size=4,
        text=text,
    )
    checkpoint_path = tmp_path / "params.pkl"

    save_checkpoint(params, metadata, checkpoint_path)
    loaded_params, loaded_metadata = load_checkpoint(checkpoint_path)

    assert len(losses) == 2
    assert losses[-1] < losses[0]
    assert checkpoint_path.exists()
    assert loaded_params.keys() == params.keys()
    assert loaded_metadata["stoi"] == metadata["stoi"]

def test_encode_decode_round_trip():
    text = "small real text"
    _, _, metadata = train_model(
        key=random.PRNGKey(5),
        epochs=1,
        block_size=4,
        batch_size=2,
        text=text * 4,
    )

    encoded = encode(text, metadata["stoi"])

    assert decode(encoded, metadata["itos"]) == text
