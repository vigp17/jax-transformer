import jax.numpy as jnp
from jax import nn

def multi_head_attention(queries, keys, values, num_heads):
    """
    Splits the work across multiple 'heads' so the AI can 
    look at multiple relationships simultaneously.
    """
    batch_size, seq_len, d_model = queries.shape
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    
    d_k = d_model // num_heads
    
    # 1. Split into heads: (batch, seq, heads, d_k)
    def split_heads(x):
        x = x.reshape((batch_size, seq_len, num_heads, d_k))
        return x.transpose((0, 2, 1, 3)) # (batch, heads, seq, d_k)

    qs, ks, vs = map(split_heads, (queries, keys, values))
    
    # 2. Scaled Dot-Product Attention for all heads at once
    scores = jnp.matmul(qs, ks.transpose((0, 1, 3, 2))) / jnp.sqrt(d_k)
    weights = nn.softmax(scores, axis=-1)
    
    # 3. Combine context from all heads
    context = jnp.matmul(weights, vs) # (batch, heads, seq, d_k)
    
    # 4. Concatenate heads back together: (batch, seq, d_model)
    context = context.transpose((0, 2, 1, 3)).reshape((batch_size, seq_len, d_model))
    
    return context, weights