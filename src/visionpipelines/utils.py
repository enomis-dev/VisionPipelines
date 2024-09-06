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
