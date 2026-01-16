import torch
import timm

def get_vit_model(model_name: str = "vit_tiny_patch16_224"):
    """Loads a ViT model from timm."""
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    return model
