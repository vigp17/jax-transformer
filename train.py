from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
from jax import random

from model.transformer import init_model_params, full_transformer

CHECKPOINT_PATH = Path("checkpoints/params.pkl")
VOCAB_SIZE = 1000
D_MODEL = 64
D_FF = 256
LEARNING_RATE = 0.05
EPOCHS = 10

# --- 1. The Error Calculator ---
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

def toy_training_data():
    """
    Creates a tiny next-token prediction example.
    """
    # Imagine ID 12="The", 45="cat", 88="sat", 9="on"
    input_tokens = jnp.array([[12, 45, 88, 9, 12]])

    # The correct next words: "cat", "sat", "on", "the", "mat" (ID: 999)
    target_tokens = jnp.array([[45, 88, 9, 12, 999]])
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
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    d_ff=D_FF,
):
    """
    Trains the tiny Transformer on the toy next-token example.
    """
    params = init_model_params(key, vocab_size, d_model, d_ff)
    input_tokens, target_tokens = toy_training_data()
    losses = []

    for _ in range(epochs):
        params, loss_value = train_step(params, input_tokens, target_tokens, learning_rate)
        losses.append(float(loss_value))

    return params, losses

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

def main():
    print(f"Running on: {jax.devices()[0]}")
    print("\n--- Starting Training Loop ---")

    params, losses = train_model()
    for epoch, loss_value in enumerate(losses, start=1):
        print(f"Epoch {epoch:2d} | Loss (Error): {loss_value:.4f}")

    save_params(params)
    print(f"\nSaved trained params to: {CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()
