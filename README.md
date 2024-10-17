# <div align = "center"> DPT Model</div>
The DPT model (Dense Prediction Transformer) is a tranformer-beased architecture for tasks like image segmentation and depth enstimation. It leverages the strength of the Vision Transformer (ViTs) foe dense prediction, achieving high accuracy in pixel predictions. Let's break it down step by step for the three cases you requested.
## Step by step process for any dense prediction task
- **Input Image** : The model takes an image as input, usually resized to a standard dimension (e.g. 224x224 or 384x384).
- **Image Tokenization** : The image is split into patches (e.g. 16x16) which are flattened into 1D vectors (tokens). Each patch gets a positional embedding to retain spatail information.
- **Transformer Encoder** : The tokens pass through multiple layers of a Vision Transformer (ViT). Each transformer layer consists of multi-head self-attention and feed-forward neural networks. The self-attention mechanism allows the model to capture long-range dependencies and relationship between different patches.
- **Feature representation** : After passing through the transformer layers, the tokens represent features of the image, with global context captured.
- **Up-sampling** : Since the transformer outputs low-resolution feature maps. the model uses up-sampling techniques (such as multi-level feature fusion or convolution layers) to convert these low-resolution features back to full image resolution.
- **Final Prediction** : A final prediction head (task-specific layer) generates the desired output. For instance, this could be pixel-wise labels in image segmentation or depth bvalues for depth estimation.

## Step by step process for Image Segmentation
- **Inout Image** : The model takes an image and preprocesses it. (e.g. resizing to 384x384)
- **Image patch Tokenization** : The image is divided into non-overlapping patches, each patch is flattened into a 1D vector, and positional embeddings are added.
- **Vision Transformer (ViT) Encoder** : The transformer encoder processes the patches through self-attention layers, building a global understanding of the image.
- **Multi-Level feature Extraction** : The segmentation task benefits from both high-level semantics features and low-level details. DPT extracts features from multiple levels of the transformer (early, middle and late layers) to capture both global context and finer details.
- **Decoder for Segementation** : The featires from the transformer are up-sampled through a decoder that gruadually increase the resilution. The decoder fuses multi-level feayures to retain fine details and spatial accuracy.
- *Pixel-wise Classification** : The up-sampled features are passed through a classification layer, where each pixel is assigned a class label (e.g. background, object etc).
- **Output (segmantation Map)** : The final output is a segmentation map, where each pixel of the image is assigned to specific category.
