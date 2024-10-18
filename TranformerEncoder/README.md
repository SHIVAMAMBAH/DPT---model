The **DPT** model uses a ViT encoder. Specifically, it incorporates the self-attention mechanim and architecture principles from the ViT model, which processes image data by diving it into patches and applying the transformer encoder on these patches.
### Structure of the encoder
The core of the DPt encoder consists of multiple transformer layers where each layer performs the following operations:
- **Multi-Head Self-Attention (MHSA)**
  - **Self-Attention** : Each patch embedding interact with all other patch embedding to capture long range dependencies. This is done using the self-attention mechanism, where each embedding attends to all others based on their pairwise similarity.
  - **Multi-Head** : Multiple attention heads are used to learn differnt types pf relationships between patches. Each head produces an output, which is then concatenated and linearly transformed.
- **Layer Normalization & skip Connections**
  - After the MHSA step, a skip connectio (residual connection) is aplied tp the original inout of the layer, and layer normalization is performed. This prevents vanishing gradient issues and helps stabilize training.
- **Feed-Forward Network (FFN)**
  - After the MHSA step, each embedding passes through a two-layer feed-forward network (FFN) with a non-linear activation function (typically GELU). This increases the representational power of the network.
  - The FFN typically consists of a fully connected layer that expands the embedding dimension (e.g., by a factor of 4), followed by a non-linear activation (GELU), and then another fully connected layer that reduces the dimension back to D.
