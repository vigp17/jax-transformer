import jax.numpy as jnp
from jax import random
from model.transformer import init_model_params, full_transformer

def test_full_transformer_shape():
    key = random.PRNGKey(3)
    vocab_size = 1000
    d_model = 64
    d_ff = 256

    input_tokens = jnp.array([[12, 45, 88, 9, 412]])
    params = init_model_params(key, vocab_size, d_model, d_ff)
    predictions = full_transformer(params, input_tokens)

    assert predictions.shape == (1, 5, vocab_size)
