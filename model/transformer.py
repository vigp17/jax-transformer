import jax.numpy as jnp
from jax import random
from model.embed import initialize_embeddings
from model.attention import init_attention_weights
from model.layers import init_ffn_weights
from model.block import transformer_block

def init_model_params(key, vocab_size, d_model, d_ff, max_seq_len=128):
    """
    Initializes every single weight and dictionary in the entire AI.
    """
    k1, k2, k3, k4, k5 = random.split(key, 5)
    
    # 1. The Word Dictionary (Phase 1)
    embeddings = initialize_embeddings(k1, vocab_size, d_model)
    positional_embeddings = random.normal(k2, (max_seq_len, d_model)) / jnp.sqrt(d_model)
    
    # 2. The Transformer Block Brain Cells (Phases 2 & 3)
    block_params = {
        "attention": init_attention_weights(k3, d_model),
        "ffn": init_ffn_weights(k4, d_model, d_ff),
    }
    
    # 3. The Final Output Layer (Translates math back into English)
    final_w = random.normal(k5, (d_model, vocab_size)) / jnp.sqrt(d_model)
    final_b = jnp.zeros((vocab_size,))
    
    return {
        "embeddings": embeddings,
        "positional_embeddings": positional_embeddings,
        "block": block_params,
        "final_w": final_w,
        "final_b": final_b
    }

def full_transformer(params, input_tokens):
    """
    The complete pipeline: ID -> Vector -> Block -> Word Prediction
    """
    # 1. Convert word IDs into Math Vectors
    seq_len = input_tokens.shape[1]
    x = params["embeddings"][input_tokens]
    x = x + params["positional_embeddings"][:seq_len]
    
    # 2. Run the math through the Transformer Block
    x = transformer_block(params["block"], x)
    
    # 3. Convert the math back into vocabulary predictions (Logits)
    logits = jnp.matmul(x, params["final_w"]) + params["final_b"]
    
    return logits
