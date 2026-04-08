from jax import random
from model.attention import init_attention_weights
from model.layers import init_ffn_weights
from model.block import transformer_block

def test_transformer_block_shape():
    key = random.PRNGKey(2)
    d_model = 64
    d_ff = 256
    batch_size = 1
    seq_length = 5

    input_data = random.normal(key, (batch_size, seq_length, d_model))
    block_params = {
        "attention": init_attention_weights(key, d_model),
        "ffn": init_ffn_weights(key, d_model, d_ff),
    }

    final_output = transformer_block(block_params, input_data)

    assert final_output.shape == input_data.shape
