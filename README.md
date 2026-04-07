# JAX Transformer from Scratch
A ground-up implementation of a Transformer Block using Google's JAX framework, optimized for research-grade performance.

## Features
- **Custom Attention:** Scaled Dot-Product Attention implemented via JAX matrix operations.
- **Hardware Agnostic:** Developed on Apple M1 (CPU/Metal) and scaled to Nvidia A100 via Google Colab.
- **From-Scratch Calculus:** Custom Cross-Entropy loss and Gradient Descent using `jax.value_and_grad`.

## Architecture
- Embedding Layer
- Multi-Head Self-Attention
- Feed-Forward Network (FFN) with ReLU activation
- Layer Normalization & Skip Connections