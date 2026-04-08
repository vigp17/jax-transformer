from jax import random
from model.layers import init_ffn_weights, feed_forward

def test_feed_forward_shape():
    key = random.PRNGKey(1)
    d_model = 64
    d_ff = 256
    batch_size = 1
    seq_length = 5

    attention_output = random.normal(key, (batch_size, seq_length, d_model))
    ffn_params = init_ffn_weights(key, d_model, d_ff)
    final_output = feed_forward(ffn_params, attention_output)

    assert final_output.shape == attention_output.shape
