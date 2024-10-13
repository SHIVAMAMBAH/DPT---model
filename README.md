# <div align = "center"> DPT Model</div>
The DPT model (Dense Prediction Transformer) is a tranformer-beased architecture for tasks like image segmentation and depth enstimation. It leverages the strength of the Vision Transformer (ViTs) foe dense prediction, achieving high accuracy in pixel predictions. Let's break it down step by step for the three cases you requested.
## Step by step process for any dense prediction task
- **Input Image** : The model takes an image as input, usually resized to a standard dimension (e.g. 224x224 or 384x384).
- **Image Tokenization** : The image is split into patches (e.g. 16x16) which are flattened into 1D vectors (tokens). Each patch gets a positional embedding to retain spatail information.
- **Transformer Encoder** : The tokens pass through multiple layers of a Vision Transformer (ViT). Each transformer layer consists of multi-head self-attention and feed-forward neural networks. The self-attention mechanism allows the model to capture long-range dependencies and relationship between different patches.
