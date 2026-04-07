import jax
import jax.numpy as jnp
from jax import random
from model.layers import init_ffn_weights
from model.block import transformer_block

print(f"Running on: {jax.devices()[0]}")

key = random.PRNGKey(2)
d_model = 64
d_ff = 256
batch_size = 1
seq_length = 5

# 1. The input sentence (as math)
input_data = random.normal(key, (batch_size, seq_length, d_model))

# 2. Initialize the FFN brain cells
ffn_params = init_ffn_weights(key, d_model, d_ff)

# 3. Pass it through the entire Transformer Block!
final_output = transformer_block(ffn_params, input_data)

print("\n--- Complete Transformer Block Test Successful! ---")
print(f"Input shape: {input_data.shape}")
print(f"Final Block Output shape: {final_output.shape}")