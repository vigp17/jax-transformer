import jax
import jax.numpy as jnp
from jax import random
from model.transformer import init_model_params, full_transformer

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

# --- 2. Setup the Environment ---
print(f"Running on: {jax.devices()[0]}")
key = random.PRNGKey(4)
vocab_size = 1000
d_model = 64
d_ff = 256

# Initialize the architecture
params = init_model_params(key, vocab_size, d_model, d_ff)

# Create a fake sentence. 
# Imagine ID 12="The", 45="cat", 88="sat", 9="on"
input_tokens = jnp.array([[12, 45, 88, 9, 12]])

# The correct next words: "cat", "sat", "on", "the", "mat" (ID: 999)
target_tokens = jnp.array([[45, 88, 9, 12, 999]])

# --- 3. The Training Loop (Gradient Descent) ---
learning_rate = 0.05

print("\n--- Starting Training Loop ---")
# We will train the AI for 10 "Epochs" (10 repetitions of studying the data)
for epoch in range(10):
    
    # JAX MAGIC: value_and_grad runs the network, calculates the error, 
    # AND performs complex calculus to find the derivative (gradient) for every single weight.
    loss_value, gradients = jax.value_and_grad(cross_entropy_loss)(params, input_tokens, target_tokens)
    
    # Update every dictionary of weights in the entire AI simultaneously
    params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g, 
        params, 
        gradients
    )
    
    print(f"Epoch {epoch + 1:2d} | Loss (Error): {loss_value:.4f}")