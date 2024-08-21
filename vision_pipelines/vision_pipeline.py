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
        image_tensor = torch.tensor(image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC to NCHW

        for task in self.tasks:
            image_tensor = task(image_tensor)

        # Convert the tensor back to numpy
        image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)  # NCHW to HWC
        image = (image_tensor * 255).byte().numpy()

        return image

    def clear_pipeline(self):
        """Clear all tasks from the pipeline."""
        self.tasks.clear()

# Example tasks using Kornia and OpenCV

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

def show_image() -> Callable[[torch.Tensor], torch.Tensor]:
    """Display the image using OpenCV."""
    def _show_image(image: torch.Tensor) -> torch.Tensor:
        image_np = image.squeeze(0).permute(1, 2, 0).byte().numpy()
        cv2.imshow('Image', image_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return image
    return _show_image

# Example usage
if __name__ == "__main__":
    # Load an example image
    image = cv2.imread('example.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Kornia

    # Create the pipeline
    pipeline = ImagePipeline()

    # Add tasks to the pipeline
    pipeline.add_task(resize((300, 300)))
    pipeline.add_task(to_grayscale())
    pipeline.add_task(apply_threshold(127))
    pipeline.add_task(show_image())  # Show image and pass it on

    # Run the pipeline
    processed_image = pipeline.run_pipeline(image)
