from typing import List, Optional, Tuple, Union
import numpy as np
import cv2
import PIL
from PIL import Image
import torch

# ? taken from hugging face diffuser :
# ? https://github.com/huggingface/diffusers/blob/82be58c51272dcc7ebd5cbf8f48d444e3df96a1a/src/diffusers/utils/pil_utils.py

# * Some Note about various lib image format  : 
# * 1. image value range
# *     (1) PIL : [0, 1] for PIL.Image.save(.)
# *     (2) Torch : [-1, 1] for norm, denorm function in this codebase
# * 2. channel & format
# *     (1) opencv use bgr, torch, pil use rgb
# *     (2) opencv, pil use channel first, torch use channel last

# * All to PIL format :
def pt_to_pil(images: torch.FloatTensor):
    """
    Convert a torch image or a batch of images to a PIL image.
    """
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = np_to_pil(images)
    return images

def np_to_pil(images: np.ndarray) -> List[PIL.Image.Image]:
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

# * All to numpy format :
def pil_to_np(images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:
    """
    Convert a PIL image or a list of PIL images to NumPy arrays.
    """
    if not isinstance(images, list):
        images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)

    return images

def pt_to_np(images: torch.FloatTensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images

# * All to torch format :
def np_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """
    Convert a NumPy images to a PyTorch tensor.
    """
    if images.ndim == 3:
        images = images[..., None]

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images

def pil_to_pt(images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> torch.FloatTensor:
    """
    Convert a PIL image or a list of PIL images to a PyTorch tensor.
    """
    if not isinstance(images, list):
        images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images

def cvnp_to_pt(image: np.ndarray) -> torch.FloatTensor:
    """
    Convert a cv2 readed image to a PyTorch tensor.
    """
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images

# * norm & denorm with range [-1, 1] <-> [0, 1] :
def normalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize an image array to [-1,1].
    """
    return 2.0 * images - 1.0

def denormalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Denormalize an image array to [0,1].
    """
    return (images / 2 + 0.5).clamp(0, 1)

def convert_to_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    Converts a PIL image to RGB format.
    """
    image = image.convert("RGB")

    return image

def convert_to_grayscale(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    Converts a PIL image to grayscale format.
    """
    image = image.convert("L")

    return image