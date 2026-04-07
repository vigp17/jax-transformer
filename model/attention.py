import jax.numpy as jnp
from jax import nn

def scaled_dot_product_attention(queries, keys, values, mask=None):
    """
    Calculates how much focus each word should give to other words.
    """
    # 1. Find the dimension size of our keys (to stabilize the math)
    d_k = queries.shape[-1]
    
    # 2. Multiply Queries by the Transpose of Keys to get raw attention scores
    scores = jnp.matmul(queries, keys.swapaxes(-1, -2)) / jnp.sqrt(d_k)
    
    # Optional: Hide future words if we are generating text
    if mask is not None:
        scores = jnp.where(mask == 0, -1e9, scores)
        
    # 3. Convert scores into percentages (probabilities) using Softmax
    attention_weights = nn.softmax(scores, axis=-1)
    
    # 4. Multiply the weights by the Values to get the final context
    output = jnp.matmul(attention_weights, values)
    
    return output, attention_weights