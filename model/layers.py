import jax.numpy as jnp
from jax import random, nn

def init_ffn_weights(key, d_model, d_ff):
    """
    Initializes the 'brain cells' (weights and biases) for the reasoning layer.
    d_model: the size of our input vector (e.g., 64)
    d_ff: the expanded size for reasoning (usually 4x larger, e.g., 256)
    """
    k1, k2 = random.split(key)
    
    # Layer 1 Weights and Biases
    w1 = random.normal(k1, (d_model, d_ff)) / jnp.sqrt(d_model)
    b1 = jnp.zeros((d_ff,))
    
    # Layer 2 Weights and Biases
    w2 = random.normal(k2, (d_ff, d_model)) / jnp.sqrt(d_ff)
    b2 = jnp.zeros((d_model,))
    
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

def feed_forward(params, x):
    """
    The actual reasoning math: Linear -> ReLU Activation -> Linear
    """
    # 1. First linear transformation (expand the vector)
    hidden = jnp.matmul(x, params["w1"]) + params["b1"]
    
    # 2. Activation Function (ReLU): This allows the AI to learn non-linear, complex concepts
    hidden = nn.relu(hidden)
    
    # 3. Second linear transformation (compress it back to original size)
    output = jnp.matmul(hidden, params["w2"]) + params["b2"]
    
    return output