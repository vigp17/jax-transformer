import jax.numpy as jnp
from jax import random

def initialize_embeddings(key, vocab_size, embedding_dim):
    """
    Creates a massive lookup table of random numbers.
    Each row represents a single word in our vocabulary.
    """
    return random.normal(key, (vocab_size, embedding_dim))