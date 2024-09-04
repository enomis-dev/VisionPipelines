import cv2
import torch
import kornia
import numpy as np
from typing import List, Callable, Union

class VisionPipeline:
    def __init__(self):
        self.tasks: List[Callable[[torch.Tensor], torch.Tensor]] = []

    def add_task(self, task: Callable[[torch.Tensor], torch.Tensor]):
        """Add a new task to the pipeline."""
        self.tasks.append(task)

    def run_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Run all tasks on the given image in sequence."""
        # Convert the image from numpy to a PyTorch tensor
        # image_tensor = torch.tensor(image).float() / 255.0
        # image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC to NCHW

        for task in self.tasks:
            image = task(image)

        # Convert the tensor back to numpy
        # image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)  # NCHW to HWC
        # image = (image_tensor * 255).byte().numpy()

        return image

    def clear_pipeline(self):
        """Clear all tasks from the pipeline."""
        self.tasks.clear()
