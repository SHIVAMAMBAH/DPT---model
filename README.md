# <div align = "center"> DPT Model</div>
The DPT model (Dense Prediction Transformer) is a tranformer-beased architecture for tasks like image segmentation and depth enstimation. It leverages the strength of the Vision Transformer (ViTs) for dense prediction, achieving high accuracy in pixel predictions. Let's break it down step by step for the three cases.
## Step by step process for any dense prediction task
- **Input Image** : The model takes an image as input, usually resized to a standard dimension (e.g. 224x224 or 384x384).
- **Image Tokenization** : The image is split into patches (e.g. 16x16) which are flattened into 1D vectors (tokens). Each patch gets a positional embedding to retain spatail information.
- **Transformer Encoder** : The tokens pass through multiple layers of a Vision Transformer (ViT). Each transformer layer consists of multi-head self-attention and feed-forward neural networks. The self-attention mechanism allows the model to capture long-range dependencies and relationship between different patches.
- **Feature representation** : After passing through the transformer layers, the tokens represent features of the image, with global context captured.
- **Up-sampling** : Since the transformer outputs low-resolution feature maps. the model uses up-sampling techniques (such as multi-level feature fusion or convolution layers) to convert these low-resolution features back to full image resolution.
- **Final Prediction** : A final prediction head (task-specific layer) generates the desired output. For instance, this could be pixel-wise labels in image segmentation or depth bvalues for depth estimation.

## Step by step process for Image Segmentation
- **Input Image** : The model takes an image and preprocesses it. (e.g. resizing to 384x384)
- **Image patch Tokenization** : The image is divided into non-overlapping patches, each patch is flattened into a 1D vector, and positional embeddings are added.
- **Vision Transformer (ViT) Encoder** : The transformer encoder processes the patches through self-attention layers, building a global understanding of the image.
- **Multi-Level feature Extraction** : The segmentation task benefits from both high-level semantics features and low-level details. DPT extracts features from multiple levels of the transformer (early, middle and late layers) to capture both global context and finer details.
- **Decoder for Segementation** : The featires from the transformer are up-sampled through a decoder that gruadually increase the resilution. The decoder fuses multi-level feayures to retain fine details and spatial accuracy.
- **Pixel-wise Classification** : The up-sampled features are passed through a classification layer, where each pixel is assigned a class label (e.g. background, object etc).
- **Output (segmantation Map)** : The final output is a segmentation map, where each pixel of the image is assigned to specific category.
## Step by step process for Depth Estimation
- **Input Image** : Similar to segmentation, the input image is resized to a standard size like 384x384.
- **Patch Tokenization** : The image is split into patches, and positional embeddings are added to ensure spatial context is retained.
- **Vision Transformer (ViT) Encoder** : The transfomer processes the tokens, learning global dependencies between different parts of the image to capture depth-related features.
- **Depth-specific Feature Exactraction** : DPT extracts multi-level feature from different transformer layers. Since depth estimation requires understanding of object sizes, distances, and perspective, features from various transformer layers are combined to capture fine-grained depth cues.
- **Up sampling for dense prediction** : The low-resolution feature maps are up-sampled through the convolution layers or other techniques to match the original image size.
- **Depth Prediction Head** : A regressive layer is used to output continuous depth value for each pixel. Ulike segmentation, this is not a classification problem but a regression problem, where the moel outputs a continuous value for depth.
- **Output (Depth Map)** : The final output is a depth map, where each pixel has a pedicted depth value (e.g. closer objects have lower values, while farther objects have higher values).

## Object Detection
For object detection, the DPT model would need to identify where objects are in an image (bounding
boxes) and classify them into different categories. This involves not only dense prediction (predicting
something at every pixel) but also learning spatial positions and object boundaries.
Step-by-Step Process:
- **Input Image**: The model takes the input image and resizes it to a standard size (e.g.,
384x384).
- **Image Tokenization (Patches)**: Like other DPT-based tasks, the image is divided into
small, non-overlapping patches (e.g., 16x16 pixels). Each patch is flattened into a 1D vector,
creating tokens. These tokens are also given positional embeddings to retain spatial
information.
- **Vision Transformer (ViT)** Encoder: The transformer encoder processes the tokens
through multiple layers of self-attention. This helps the model understand global
dependencies in the image, such as the relationship between different parts of objects and
the surrounding context.
- **Multi-Level Feature Representation**: Object detection requires both fine-grained
details for small objects and high-level context for large objects. The DPT extracts features
from multiple transformer layers to capture this multi-scale information.
- **Object Query Generation**: In many transformer-based object detection models (like
DETR), object queries are used. The DPT would similarly generate a fixed number of object
queries (learnable embeddings). These queries interact with the image tokens through
self-attention to detect potential objects.
- **Bounding Box Regression & Classification**:
  - **Bounding Box Prediction**: For each object query, the model predicts a bounding
box (coordinates of the top-left and bottom-right corners).
  - **Object Classification**: Each object query also predicts the class label of the object
inside the bounding box (e.g., person, car, dog, etc.). This is done using a classification layer that maps the feature representation to one of the predefined
classes.
- **Multi-Level Decoder for Object Detection**: The features extracted from the transformer
are passed through a decoder to refine the bounding box predictions and class labels. Since
the DPT uses multi-level feature extraction, it can detect both large and small objects with
better accuracy.
- **Final Output (Object Detection)**: The final output consists of:
  -  A set of bounding boxes for detected objects.
  -  Class labels for each detected object.
  -  Confidence scores indicating how likely the detection is correct.
