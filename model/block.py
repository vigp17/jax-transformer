import jax.numpy as jnp
from model.attention import multi_head_attention
from model.layers import feed_forward

def layer_norm(x, eps=1e-5):
    """
    Grading on a curve: Keeps the numbers perfectly balanced.
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(variance + eps)

def transformer_block(params, x, num_heads=4):
    """
    The complete Lego brick: Attention -> Add & Norm -> Reasoning -> Add & Norm
    """
    # 1. Self-Attention (The Brain)
    # We pass 'x' in as Queries, Keys, and Values because the words are looking at themselves
    attention_out, _ = multi_head_attention(params["attention"], x, num_heads)
    
    # 2. First Skip Connection & Normalization
    x_norm1 = layer_norm(x + attention_out)
    
    # 3. Feed-Forward Network (The Reasoning)
    ffn_out = feed_forward(params["ffn"], x_norm1)
    
    # 4. Second Skip Connection & Normalization
    output = layer_norm(x_norm1 + ffn_out)
    
    return output
