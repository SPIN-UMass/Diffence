import torch
from vit_pytorch import ViT

v = ViT(
    image_size = 32,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)


def ViT_Base(num_classes=10):
    return  ViT(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 384,
    depth = 7,
    heads = 12,
    mlp_dim = 384,
    dropout = 0.0,
    emb_dropout = 0.0
)