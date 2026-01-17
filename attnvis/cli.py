import typer
from PIL import Image
import torch
from torchvision import transforms
from attnvis.vit.model import get_vit_model
from attnvis.vit.extract import VitAttentionExtractor, compute_rollout
from attnvis.vit.viz import visualize_cls_attention
import matplotlib.pyplot as plt
import os

app = typer.Typer(no_args_is_help=True)

@app.command()
def rollout(
    image_path: str,
    model_name: str = "vit_tiny_patch16_224",
    output_dir: str = "outputs",
    head_fusion: str = "mean"
):
    """Visualise ViT attention rollout."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_vit_model(model_name).to(device)
    extractor = VitAttentionExtractor(model)
    
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    attentions = extractor.extract(input_tensor)
    # Remove batch dim
    attentions = [a[0] for a in attentions]
    
    rollout_map = compute_rollout(attentions, head_fusion=head_fusion)
    
    # Visualise rollout (it's one "head")
    # Wrap it to match visualize_cls_attention expected shape (1, q_len, k_len)
    fig = visualize_cls_attention(rollout_map.unsqueeze(0))
    fig.axes[0].set_title(f"Rollout ({head_fusion})")
    
    output_path = os.path.join(output_dir, f"vit_rollout_{head_fusion}.png")
    fig.savefig(output_path)
    typer.echo(f"Saved rollout visualisation to {output_path}")

@app.command()
def swin():
    """Visualise Swin attention."""
    typer.echo("Swin not implemented yet.")

@app.command(name="vit")
def vit(
    image_path: str,
    model_name: str = "vit_tiny_patch16_224",
    output_dir: str = "outputs",
    layer: int = -1
):
    """Visualise ViT attention."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_vit_model(model_name).to(device)
    extractor = VitAttentionExtractor(model)
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Extract attention
    attentions = extractor.extract(input_tensor)
    
    # Visualise
    if layer >= len(attentions) or layer < -len(attentions):
        typer.echo(f"Layer {layer} not found. Model has {len(attentions)} layers.")
        raise typer.Exit(1)
        
    attn_map = attentions[layer][0] # Get first batch
    fig = visualize_cls_attention(attn_map)
    
    output_path = os.path.join(output_dir, f"vit_attn_layer_{layer}.png")
    fig.savefig(output_path)
    typer.echo(f"Saved visualisation to {output_path}")

if __name__ == "__main__":
    app()
