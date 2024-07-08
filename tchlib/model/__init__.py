# Note : take from solo-learn : https://github.com/vturrisi/solo-learn
# most of model constructed by timm, doc : https://timm.fast.ai/
from .convnext import convnext_tiny, convnext_small, convnext_base, convnext_large
from .resnet import resnet18, resnet50
from .wide_resnet import wide_resnet28w2, wide_resnet28w8

from .vit import vit_tiny, vit_small, vit_base, vit_large
from .swin import swin_tiny, swin_small, swin_base, swin_large

from .mlpmixer import mlpmixer_small, mlpmixer_base, mlpmixer_large
from .poolformer import (
    poolformer_s12,
    poolformer_s24,
    poolformer_s36,
    poolformer_m36,
    poolformer_m48,
)

__all__ = [
    # conv-based 
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "resnet18",
    "resnet50",
    "wide_resnet28w2",
    "wide_resnet28w8",
    # vit-based 
    "vit_tiny",
    "vit_small",
    "vit_base",
    "vit_large",
    "swin_tiny",
    "swin_small",
    "swin_base",
    "swin_large",
    # pseudo-vit based
    "mlpmixer_small",
    "mlpmixer_base",
    "mlpmixer_large",
    "poolformer_s12",
    "poolformer_s24",
    "poolformer_s36",
    "poolformer_m36",
    "poolformer_m48"
]