from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
from jax import random

from model.transformer import init_model_params, full_transformer

CHECKPOINT_PATH = Path("checkpoints/params.pkl")
DATA_PATH = Path("data/input.txt")
D_MODEL = 64
D_FF = 256
LEARNING_RATE = 0.01
TRAIN_STEPS = 100
BLOCK_SIZE = 64
BATCH_SIZE = 16

def cross_entropy_loss(params, input_tokens, target_tokens):
    """
    Calculates how mathematically wrong the AI's predictions are.
    """
    # 1. The AI makes its raw guesses (Logits)
    logits = full_transformer(params, input_tokens)
    
    # 2. Convert the correct answers into math (One-Hot Encoding)
    vocab_size = logits.shape[-1]
    targets_one_hot = jax.nn.one_hot(target_tokens, vocab_size)
    
    # 3. Convert the AI's guesses into probabilities
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    
    # 4. Compare them to get the final Error (Loss) score
    loss = -jnp.sum(targets_one_hot * log_probs) / input_tokens.size
    return loss

def load_text(path=DATA_PATH):
    """
    Loads plain text for language-model training.
    """
    return Path(path).read_text(encoding="utf-8")

def build_vocab(text):
    """
    Builds a character vocabulary from the training text.
    """
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(text, stoi):
    """
    Converts text into token IDs.
    """
    return jnp.array([stoi[ch] for ch in text], dtype=jnp.int32)

def decode(token_ids, itos):
    """
    Converts token IDs back into text.
    """
    return "".join(itos[int(token_id)] for token_id in token_ids)

def make_batch(data, block_size, batch_size, step):
    """
    Creates deterministic next-character batches from real text.
    """
    max_start = data.shape[0] - block_size - 1
    if max_start <= 0:
        raise ValueError("Training text must be longer than block_size + 1 characters.")

    starts = (jnp.arange(batch_size) + step * batch_size) % max_start
    positions = starts[:, None] + jnp.arange(block_size)[None, :]
    input_tokens = data[positions]
    target_tokens = data[positions + 1]
    return input_tokens, target_tokens

@jax.jit
def train_step(params, input_tokens, target_tokens, learning_rate):
    """
    Runs one compiled gradient descent step.
    """
    loss_value, gradients = jax.value_and_grad(cross_entropy_loss)(
        params, input_tokens, target_tokens
    )
    params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g,
        params,
        gradients,
    )
    return params, loss_value

def train_model(
    key=random.PRNGKey(4),
    epochs=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    d_model=D_MODEL,
    d_ff=D_FF,
    block_size=BLOCK_SIZE,
    batch_size=BATCH_SIZE,
    text=None,
    data_path=DATA_PATH,
):
    """
    Trains a tiny character-level Transformer on real text.
    """
    text = load_text(data_path) if text is None else text
    stoi, itos = build_vocab(text)
    data = encode(text, stoi)
    params = init_model_params(
        key,
        vocab_size=len(stoi),
        d_model=d_model,
        d_ff=d_ff,
        max_seq_len=block_size,
    )
    losses = []

    for step in range(epochs):
        input_tokens, target_tokens = make_batch(data, block_size, batch_size, step)
        params, loss_value = train_step(params, input_tokens, target_tokens, learning_rate)
        losses.append(float(loss_value))

    metadata = {
        "stoi": stoi,
        "itos": itos,
        "block_size": block_size,
        "d_model": d_model,
        "d_ff": d_ff,
    }
    return params, losses, metadata

def save_params(params, path=CHECKPOINT_PATH):
    """
    Saves model parameters so generation does not need to retrain.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = jax.tree_util.tree_map(jax.device_get, params)
    with path.open("wb") as f:
        pickle.dump(params, f)

def load_params(path=CHECKPOINT_PATH):
    """
    Loads saved model parameters.
    """
    with Path(path).open("rb") as f:
        return pickle.load(f)

def save_checkpoint(params, metadata, path=CHECKPOINT_PATH):
    """
    Saves model parameters plus the text vocabulary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = jax.tree_util.tree_map(jax.device_get, params)
    with path.open("wb") as f:
        pickle.dump({"params": params, "metadata": metadata}, f)

def load_checkpoint(path=CHECKPOINT_PATH):
    """
    Loads model parameters plus the text vocabulary.
    """
    with Path(path).open("rb") as f:
        checkpoint = pickle.load(f)

    if "params" not in checkpoint or "metadata" not in checkpoint:
        raise ValueError("Checkpoint is missing params or metadata. Run train.py again.")

    return checkpoint["params"], checkpoint["metadata"]

def main():
    print(f"Running on: {jax.devices()[0]}")
    print(f"Training on: {DATA_PATH}")
    print("\n--- Starting Training Loop ---")

    params, losses, metadata = train_model()
    for epoch, loss_value in enumerate(losses, start=1):
        if epoch == 1 or epoch % 10 == 0:
            print(f"Step {epoch:3d} | Loss (Error): {loss_value:.4f}")

    save_checkpoint(params, metadata)
    print(f"\nSaved trained params to: {CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()
