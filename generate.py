import jax.numpy as jnp
from jax import random
from model.transformer import full_transformer
from train import CHECKPOINT_PATH, decode, encode, load_checkpoint, save_checkpoint, train_model

def predict_next_token(params, context, key, temperature):
    """
    Samples one next token from the model's final-position logits.
    """
    logits = full_transformer(params, context)
    logits = logits[:, -1, :] / temperature
    return random.categorical(key, logits, axis=-1)

def generate_tokens(params, start_tokens, length=80, block_size=64, temperature=0.8, key=random.PRNGKey(0)):
    """
    Generates token IDs by repeatedly predicting the next character.
    """
    generated = start_tokens
    
    print("AI is thinking...")
    for _ in range(length):
        key, sample_key = random.split(key)
        # 1. Get predictions for the current sequence
        context = generated[:, -block_size:]
        next_token = predict_next_token(params, context, sample_key, temperature)
        
        # 3. Add the new word to our sequence and repeat
        next_token = next_token.reshape((1, 1))
        generated = jnp.concatenate([generated, next_token], axis=1)
        
    return generated

def generate_text(params, prompt, metadata, length=100):
    """
    Generates text from a text prompt.
    """
    stoi = metadata["stoi"]
    itos = metadata["itos"]
    block_size = metadata["block_size"]
    prompt = "".join(ch for ch in prompt if ch in stoi)
    if not prompt:
        prompt = next(iter(stoi))

    start_tokens = encode(prompt, stoi).reshape((1, -1))
    if start_tokens.shape[1] < block_size:
        pad_width = block_size - start_tokens.shape[1]
        pad_token = start_tokens[:, :1]
        start_tokens = jnp.concatenate(
            [jnp.repeat(pad_token, pad_width, axis=1), start_tokens],
            axis=1,
        )

    generated = generate_tokens(params, start_tokens, length=length, block_size=block_size)
    visible_tokens = generated[0, -length - len(prompt):]
    return decode(visible_tokens, itos)

if __name__ == "__main__":
    if CHECKPOINT_PATH.exists():
        params, metadata = load_checkpoint()
        print(f"Loaded trained params from: {CHECKPOINT_PATH}")
    else:
        print("No saved params found. Training once before generation...")
        params, _, metadata = train_model()
        save_checkpoint(params, metadata)
        print(f"Saved trained params to: {CHECKPOINT_PATH}")
    
    result = generate_text(params, prompt="The ", metadata=metadata, length=80)
    print(f"\nGenerated text:\n{result}")
