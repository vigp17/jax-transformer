import jax
import jax.numpy as jnp
from jax import random
from model.layers import init_ffn_weights, feed_forward

print(f"Running on: {jax.devices()[0]}")

# 1. Setup random key and dimensions (matching our previous test)
key = random.PRNGKey(1)
d_model = 64   # Size of the vector
d_ff = 256     # The expanded "reasoning" size

batch_size = 1
seq_length = 5

# 2. Simulate the data coming out of our Attention layer
attention_output = random.normal(key, (batch_size, seq_length, d_model))

# 3. Initialize the 'brain cells' (weights and biases) for the FFN
ffn_params = init_ffn_weights(key, d_model, d_ff)

# 4. Run the data through the reasoning layer
final_output = feed_forward(ffn_params, attention_output)

print("\n--- Reasoning Layer Test Successful! ---")
print(f"Input shape from Attention: {attention_output.shape}")
print(f"Output shape after Reasoning: {final_output.shape}")