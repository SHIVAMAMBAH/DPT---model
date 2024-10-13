import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# Self-Attention mechanism for image patches
class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SimpleSelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # x: (batch_size, num_patches, embed_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # (batch_size, num_patches, num_patches)
        attention_weights = self.softmax(attention_scores)  # Normalize attention scores
        
        attention_output = torch.bmm(attention_weights, V)
        return attention_output, attention_weights

# Function to visualize attention
def visualize_attention(image_path, patch_size=32, embed_dim=64):
    # Step 1: Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    
    # Step 2: Divide the image into patches
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()  # Convert to numpy for visualization
    batch_size, channels, height, width = img_tensor.shape
    patches = img_tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(batch_size, channels, -1, patch_size, patch_size)  # (1, 3, num_patches, 32, 32)
    num_patches = patches.shape[2]
    
    # Flatten patches into vectors (each patch has shape patch_size*patch_size*channels = 32*32*3 = 3072)
    patch_vectors = patches.view(batch_size, channels, num_patches, -1).permute(0, 2, 1, 3).reshape(batch_size, num_patches, -1)
    
    # Step 3: Add a linear layer to project patch vectors to the embedding dimension
    linear_proj = nn.Linear(patch_vectors.shape[-1], embed_dim)
    patch_vectors_proj = linear_proj(patch_vectors)  # Project patch vectors to (batch_size, num_patches, embed_dim)
    
    # Step 4: Apply self-attention to the patches
    attention = SimpleSelfAttention(embed_dim)
    attention_output, attention_weights = attention(patch_vectors_proj)
    
    # Step 5: Average the attention weights across heads (for visualization)
    avg_attention = attention_weights.mean(dim=1).squeeze(0).detach().numpy()  # (num_patches,)
    
    # Reshape attention map to the patch grid
    attention_map = avg_attention.reshape(int(height / patch_size), int(width / patch_size))
    
    # Step 6: Visualize the attention map
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original image
    ax[0].imshow(img_np)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    # Attention heatmap
    ax[1].imshow(img_np)
    ax[1].imshow(attention_map, cmap='jet', alpha=0.6, extent=(0, 224, 224, 0))  # Overlay with transparency
    ax[1].set_title("Attention Map Overlay")
    ax[1].axis('off')
    
    plt.show()

# Provide the image path here
img_path = 'path_to_your_image.jpg'
visualize_attention(img_path)

