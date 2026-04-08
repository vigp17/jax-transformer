from jax import random
from model.attention import init_attention_weights, multi_head_attention

def test_multi_head_attention_shapes_and_causal_mask():
    key = random.PRNGKey(0)
    batch_size = 1
    seq_length = 5
    d_model = 64
    num_heads = 4

    x = random.normal(key, (batch_size, seq_length, d_model))
    attention_params = init_attention_weights(key, d_model)

    output, weights = multi_head_attention(attention_params, x, num_heads)

    assert output.shape == (batch_size, seq_length, d_model)
    assert weights.shape == (batch_size, num_heads, seq_length, seq_length)
    assert weights[0, 0, 0, 1] == 0.0
