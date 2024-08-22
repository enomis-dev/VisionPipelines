import cv2
import torch
import kornia
import numpy as np
from typing import List, Callable, Union


def resize(size: tuple) -> Callable[[torch.Tensor], torch.Tensor]:
    """Resize the image to the given size using Kornia."""
    def _resize(image: torch.Tensor) -> torch.Tensor:
        return kornia.geometry.transform.resize(image, size)
    return _resize

def to_grayscale() -> Callable[[torch.Tensor], torch.Tensor]:
    """Convert the image to grayscale using Kornia."""
    def _to_grayscale(image: torch.Tensor) -> torch.Tensor:
        return kornia.color.rgb_to_grayscale(image)
    return _to_grayscale

def apply_threshold(threshold: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Apply a binary threshold to the image using OpenCV."""
    def _apply_threshold(image: torch.Tensor) -> torch.Tensor:
        image_np = image.squeeze(0).permute(1, 2, 0).byte().numpy()
        _, result = cv2.threshold(image_np, threshold, 255, cv2.THRESH_BINARY)
        return torch.tensor(result).float().unsqueeze(0).permute(2, 0, 1) / 255.0
    return _apply_threshold