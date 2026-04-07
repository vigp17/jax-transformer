import jax
import jax.numpy as jnp
from jax import random
from model.attention import scaled_dot_product_attention

# 1. Verify hardware
print(f"Running on: {jax.devices()[0]}")

# 2. Set up a random number generator
key = random.PRNGKey(0)

# 3. Create fake data for Queries, Keys, and Values
# Imagine this is a sentence with 5 words, and each word is a 64-dimension vector
batch_size = 1
seq_length = 5
d_model = 64

q = random.normal(key, (batch_size, seq_length, d_model))
k = random.normal(key, (batch_size, seq_length, d_model))
v = random.normal(key, (batch_size, seq_length, d_model))

# 4. Run the math through our Attention function
output, weights = scaled_dot_product_attention(q, k, v)

print("\n--- Attention Test Successful! ---")
print(f"Output shape (Matches input): {output.shape}")
print(f"Attention Weights shape (Word-to-Word matrix): {weights.shape}")