import torch
import unittest
from attnvis.vit.model import get_vit_model
from attnvis.vit.extract import VitAttentionExtractor

class TestViTExtraction(unittest.TestCase):
    def test_extraction_shapes(self):
        model_name = "vit_tiny_patch16_224"
        model = get_vit_model(model_name)
        extractor = VitAttentionExtractor(model)
        
        x = torch.randn(1, 3, 224, 224)
        attentions = extractor.extract(x)
        
        # vit_tiny has 12 layers
        self.assertEqual(len(attentions), 12)
        
        # vit_tiny has 3 heads
        # 224/16 = 14. 14*14 = 196. +1 CLS = 197 tokens.
        for attn in attentions:
            self.assertEqual(attn.shape, (1, 3, 197, 197))

if __name__ == "__main__":
    unittest.main()
