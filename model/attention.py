import jax.numpy as jnp
from jax import nn, random

def init_attention_weights(key, d_model):
    """
    Initializes trainable projections for multi-head self-attention.
    """
    kq, kk, kv, ko = random.split(key, 4)
    scale = jnp.sqrt(d_model)
    return {
        "wq": random.normal(kq, (d_model, d_model)) / scale,
        "wk": random.normal(kk, (d_model, d_model)) / scale,
        "wv": random.normal(kv, (d_model, d_model)) / scale,
        "wo": random.normal(ko, (d_model, d_model)) / scale,
    }

def scaled_dot_product_attention(queries, keys, values, causal=True):
    """
    Computes scaled dot-product attention for tensors with shape
    (batch, heads, seq, d_k).
    """
    seq_len = queries.shape[-2]
    d_k = queries.shape[-1]
    scores = jnp.matmul(queries, keys.transpose((0, 1, 3, 2))) / jnp.sqrt(d_k)

    if causal:
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        scores = jnp.where(mask, scores, -1e9)

    weights = nn.softmax(scores, axis=-1)
    return jnp.matmul(weights, values), weights

def multi_head_attention(params, x, num_heads, causal=True):
    """
    Splits the work across multiple 'heads' so the AI can 
    look at multiple relationships simultaneously.
    """
    batch_size, seq_len, d_model = x.shape
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    
    d_k = d_model // num_heads
    queries = jnp.matmul(x, params["wq"])
    keys = jnp.matmul(x, params["wk"])
    values = jnp.matmul(x, params["wv"])
    
    # 1. Split into heads: (batch, seq, heads, d_k)
    def split_heads(x):
        x = x.reshape((batch_size, seq_len, num_heads, d_k))
        return x.transpose((0, 2, 1, 3)) # (batch, heads, seq, d_k)

    qs, ks, vs = map(split_heads, (queries, keys, values))
    
    # 2. Scaled Dot-Product Attention for all heads at once
    context, weights = scaled_dot_product_attention(qs, ks, vs, causal=causal)
    
    # 3. Concatenate heads back together and apply the output projection
    context = context.transpose((0, 2, 1, 3)).reshape((batch_size, seq_len, d_model))
    context = jnp.matmul(context, params["wo"])
    
    return context, weights
