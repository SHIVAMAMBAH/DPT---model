The **DPT** model uses a ViT encoder. Specifically, it incorporates the self-attention mechanim and architecture principles from the ViT model, which processes image data by diving it into patches and applying the transformer encoder on these patches.
### Structure of the encoder
The core of the DPt encoder consists of multiple transformer layers where each layer performs the following operations:
- **Multi-Head Self-Attention (MHSA)**
  - **Self-Attention** : Each patch embedding interact with all other patch embedding to capture long range dependencies. This is done using the self-attention mechanism, where each embedding attends to all others based on their pairwise similarity.
