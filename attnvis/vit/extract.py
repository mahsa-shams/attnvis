import torch

class VitAttentionExtractor:
    """Helper to extract attention maps from ViT models."""
    def __init__(self, model):
        self.model = model
        self.attentions = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if hasattr(module, 'fused_attn'):
                module.fused_attn = False
            
            # In timm, the Attention module usually has an 'attn_drop' 
            # but that might be an Identity if dropout is 0.
            # However, it still exists.
            if name.endswith('attn.attn_drop'):
                module.register_forward_hook(self._get_hook(name))

    def _get_hook(self, name):
        def hook(module, input, output):
            # The input to attn_drop is the attention matrix after softmax
            self.attentions.append(output)
        return hook

    def extract(self, x):
        self.attentions = []
        _ = self.model(x)
        return self.attentions

def compute_rollout(attentions, discard_ratio=0.9, head_fusion='mean'):
    """
    Computes attention rollout.
    attentions: list of attention maps, each (num_heads, q_len, k_len)
    """
    result = torch.eye(attentions[0].shape[-1])
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == 'mean':
                attention_heads_fused = attention.mean(axis=0)
            elif head_fusion == 'max':
                attention_heads_fused = attention.max(axis=0)[0]
            elif head_fusion == 'min':
                attention_heads_fused = attention.min(axis=0)[0]
            else:
                raise ValueError(f"Invalid head_fusion: {head_fusion}")

            # Flat the attention
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            # flat[0, indices] = 0 # This might be wrong if we want to keep high values

            # Add identity to account for residual connections
            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + I) / 2
            a = a / a.sum(dim=-1, keepdim=True)

            result = torch.matmul(a, result)
    
    return result
