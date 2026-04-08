import jax.numpy as jnp
from model.transformer import full_transformer
from train import CHECKPOINT_PATH, load_params, save_params, train_model

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

if __name__ == "__main__":
    if CHECKPOINT_PATH.exists():
        params = load_params()
        print(f"Loaded trained params from: {CHECKPOINT_PATH}")
    else:
        print("No saved params found. Training once before generation...")
        params, _ = train_model()
        save_params(params)
        print(f"Saved trained params to: {CHECKPOINT_PATH}")
    
    # Start with the word "The" (ID 12)
    start_context = jnp.array([[12]])
    
    result = generate_text(params, start_context, length=4)
    print(f"\nGenerated sequence of IDs: {result}")
