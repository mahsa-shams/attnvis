import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

def visualize_cls_attention(attention_map, image_size=(224, 224), patch_size=16):
    """
    Visualizes attention from the CLS token to the rest of the patches.
    attention_map: shape (num_heads, q_len, k_len)
    """
    num_heads = attention_map.shape[0]
    # CLS token is at index 0
    cls_attention = attention_map[:, 0, 1:]  # (num_heads, num_patches)
    
    # Calculate grid size
    grid_size = image_size[0] // patch_size
    
    fig, axes = plt.subplots(1, num_heads, figsize=(num_heads * 4, 4))
    if num_heads == 1:
        axes = [axes]
        
    for i in range(num_heads):
        head_attn = cls_attention[i].reshape(grid_size, grid_size).detach().cpu().numpy()
        
        # Normalize to [0, 1] for better visualization
        head_min = head_attn.min()
        head_max = head_attn.max()
        if head_max > head_min:
            head_attn = (head_attn - head_min) / (head_max - head_min)
        
        axes[i].imshow(head_attn, cmap='viridis', extent=(0, image_size[0], image_size[1], 0), interpolation='bilinear')
        axes[i].axis('off')
        axes[i].set_title(f"Head {i}")
    
    return fig
