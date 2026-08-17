import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from visionpipelines.tasks.registration_task import RegistrationTask
from visionpipelines.constants import RegistrationMethod
from visionpipelines.pipelines.vision_pipeline import TaskBasedPipeline


class RegistrationPipeline(TaskBasedPipeline):
    """
    Pipeline for image registration (alignment).
    
    This pipeline registers two images by detecting keypoints, matching them,
    and computing a transformation to align one image with the other.
    """
    
    def __init__(self, method: RegistrationMethod):
        """
        Initialize the registration pipeline.

        Args:
            method: The registration method to use (RegistrationMethod enum).
        """
        task = RegistrationTask(method=method)
        super().__init__(task=task)
        self.registrator = task  # Alias for direct task access

    def run_pipeline(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the registration pipeline on two images.
        
        Args:
            image1: The reference image.
            image2: The image to be registered to image1.
            
        Returns:
            Tuple of (registered_image, keypoints) where:
            - registered_image: image2 warped to align with image1
            - keypoints: Array of shape (4, N) containing matched keypoint coordinates
        """
        return super().run_pipeline(image1, image2)

    def plot_matches(self, image1: np.ndarray, image2: np.ndarray, keypoints: np.ndarray):
        """
        Plot the matches between two images.

        Args:
            image1: First image (as a numpy array, RGB or grayscale).
            image2: Second image (as a numpy array, RGB or grayscale).
            keypoints: Array of keypoints with shape (4, n), where:
                keypoints[0, :] - x coordinates for image1
                keypoints[1, :] - y coordinates for image1
                keypoints[2, :] - x coordinates for image2
                keypoints[3, :] - y coordinates for image2
        """
        # Create a combined image by stacking the two images horizontally
        combined_image = np.hstack((image1, image2))

        # Plot the combined image
        plt.figure(figsize=(10, 5))
        if len(combined_image.shape) == 2:
            plt.imshow(combined_image, cmap='gray')
        else:
            plt.imshow(combined_image)

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
