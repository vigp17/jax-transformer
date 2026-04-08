# JAX Multi-Head Transformer from Scratch
A high-performance, research-oriented implementation of the Transformer architecture built using **Google JAX** and **XLA** optimization.

## 🚀 Project Overview
This repository contains a ground-up implementation of the Transformer block, designed to demonstrate the transition from local prototyping on **Apple Silicon (M1)** to cloud-scale training on **Nvidia A100 GPUs**. Unlike standard implementations, this project focuses on mathematical transparency and hardware-agnostic performance.

### Key Engineering Features:
* **Vectorized Multi-Head Attention:** Implemented 4-head causal attention using JAX's functional paradigm for maximum parallelization.
* **XLA Optimized:** Leveraged Just-In-Time (JIT) compilation to fuse kernels, reducing overhead during high-dimensional matrix multiplications.
* **Hardware Agnostic:** Seamlessly switches between `Metal` (macOS) and `CUDA` (Linux/Colab) backends.
* **Interpretability Suite:** Built-in visualization tools to extract and map internal attention weights.

## 🏗 Architecture
The model follows the original *Attention is All You Need* blueprint with modern optimizations:
1.  **Embedding Layer:** Learned weight matrix for token-to-vector mapping.
2.  **Transformer Block:** * Multi-Head Self-Attention (MHA)
    * Layer Normalization (post-norm architecture)
    * Feed-Forward Network (FFN) with ReLU activation
    * Skip Connections to mitigate vanishing gradients.
3.  **Linear Head:** Final projection to vocabulary size for next-token prediction.

## 📊 Performance & Interpretability
The model was benchmarked on a Google Colab Pro **Nvidia A100 GPU**:
* **Training Time:** ~12.14 seconds for 10 epochs.
* **Convergence:** Initial Loss: 7.41 → Final Loss: 0.56.

### Attention Heatmap Analysis
By visualizing the attention heads, we can observe the model's internal reasoning:
* **Local Focus:** Some heads concentrate on the diagonal, learning immediate syntactic relationships.
* **Global Context:** Other heads bridge longer-range dependencies, identifying repeated tokens across the sequence.

![Attention Heatmap](notebooks/attention_heatmap.png)

## 🛠 How to Run
### Local Setup (macOS/Linux)
1. Clone the repo: `git clone https://github.com/vigp17/jax-transformer.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py`
4. Generate text: `python generate.py`
5. Run tests: `python -m pytest -q`

### Cloud Scaling
Open `notebooks/JAX_Transformer_Training_A100.ipynb` in Google Colab to run on T4/A100 GPUs.
