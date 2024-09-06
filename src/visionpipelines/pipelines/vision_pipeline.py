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
        for task in self.tasks:
            image = task(image)
        return image

    def clear_pipeline(self):
        """Clear all tasks from the pipeline."""
        self.tasks.clear()
