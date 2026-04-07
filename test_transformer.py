import jax
import jax.numpy as jnp
from jax import random
from model.transformer import init_model_params, full_transformer

print(f"Running on: {jax.devices()[0]}")

key = random.PRNGKey(3)
vocab_size = 1000  # Our AI knows 1,000 words
d_model = 64
d_ff = 256
batch_size = 1
seq_length = 5

# 1. Create a fake sentence. Let's pretend the words have ID numbers: [12, 45, 88, 9, 412]
input_tokens = jnp.array([[12, 45, 88, 9, 412]])

# 2. Initialize the entire AI model
params = init_model_params(key, vocab_size, d_model, d_ff)

# 3. Ask the AI to predict the next words
predictions = full_transformer(params, input_tokens)

print("\n--- Full Transformer Pipeline Successful! ---")
print(f"Input Sentence shape (5 words): {input_tokens.shape}")
print(f"Final Predictions shape: {predictions.shape}")