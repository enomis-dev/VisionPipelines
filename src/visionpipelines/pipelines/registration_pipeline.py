import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from visionpipelines.tasks.registration_task import RegistrationTask
from visionpipelines.constants import DetectionMethod
from visionpipelines.pipelines.vision_pipeline import VisionPipeline


class RegistrationPipeline(VisionPipeline):
    def __init__(
        self, method: str, model: torch.nn.Module = None,
        device: torch.device = torch.device('cpu')):
        """Initialize the registration pipeline with a registration model."""
        super().__init__()
        self.registrator = RegistrationTask(method=method)

    def register_images(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect objects in the image using the specified method."""
        registered_image, keypoints = self.registrator.register_images(image1, image2)
        return registered_image, keypoints

    def pre_process(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        image1, image2 = self.registrator.pre_process(image1, image2)
        return image1, image2

    def run_pipeline(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """Perform object detection"""
        image1, image2 = self.pre_process(image1, image2)
        registered_image, keypoints = self.register_images(image1, image2)
        return registered_image, keypoints

    def draw_boxes(self, image: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Draw bounding boxes on the image."""
        return self.registrator.draw_boxes(image, boxes)

    def plot_matches(self, image1, image2, keypoints):
        """
        Plots the matches between two images.

        Parameters:
        - image1: First image (as a numpy array)
        - image2: Second image (as a numpy array)
        - keypoints: A list or array of keypoints with shape (4, n), where:
            keypoints[0, :] - x coordinates for image1
            keypoints[1, :] - y coordinates for image1
            keypoints[2, :] - x coordinates for image2
            keypoints[3, :] - y coordinates for image2
        """
        # Create a combined image by stacking the two images horizontally
        combined_image = np.hstack((image1, image2))

        # Plot the combined image
        plt.figure(figsize=(10, 5))
        plt.imshow(combined_image, cmap='gray')

        num_keypoints = keypoints.shape[1]

        # Plot each matched pair of keypoints
        for i in range(num_keypoints):
            # Coordinates for keypoints in image1
            x1, y1 = keypoints[0, i], keypoints[1, i]
            # Coordinates for keypoints in image2 (shifted by the width of image1)
            x2, y2 = keypoints[2, i] + image1.shape[1], keypoints[3, i]

            # Plot lines and points
            plt.plot([x1, x2], [y1, y2], color='yellow', linewidth=0.5)
            plt.scatter([x1, x2], [y1, y2], color='red', s=10)

        plt.title('Matched Keypoints Between The Two Images')
        plt.axis('off')
        plt.show()
