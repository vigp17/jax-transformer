import jax
import jax.numpy as jnp
from model.transformer import full_transformer

def generate_text(params, start_tokens, length=5):
    """
    Takes a starting word and asks the AI to keep predicting.
    """
    generated = start_tokens
    
    print("AI is thinking...")
    for _ in range(length):
        # 1. Get predictions for the current sequence
        logits = full_transformer(params, generated)
        
        # 2. Pick the word with the highest score (the last word in the sequence)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        
        # 3. Add the new word to our sequence and repeat
        next_token = next_token.reshape((1, 1))
        generated = jnp.concatenate([generated, next_token], axis=1)
        
    return generated

# --- Test the Generator ---
# We need to use the 'params' we just trained in train.py. 
# For this quick test, let's pretend we have them loaded.
# (In a real scenario, we would save the params to a file and load them here).

if __name__ == "__main__":
    from train import params, vocab_size # This imports your trained brain!
    
    # Start with the word "The" (ID 12)
    start_context = jnp.array([[12]])
    
    result = generate_text(params, start_context, length=4)
    print(f"\nGenerated sequence of IDs: {result}")