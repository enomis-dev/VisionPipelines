import torch
import numpy as np
from abc import ABC, abstractmethod

# Abstract base class for all tasks in the pipeline
class Task(ABC):
    @abstractmethod
    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Process the input image tensor and return the modified tensor."""
        pass
