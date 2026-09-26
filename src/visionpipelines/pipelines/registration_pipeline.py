import numpy as np
import matplotlib.pyplot as plt
from visionpipelines.tasks.registration_task import RegistrationResult, RegistrationTask
from visionpipelines.constants import RegistrationMethod
from visionpipelines.pipelines.vision_pipeline import TaskBasedPipeline


class RegistrationPipeline(TaskBasedPipeline):
    """
    Pipeline for image registration (alignment).
    
    This pipeline registers two images by detecting keypoints, matching them,
    computing a homography, and warping one image onto the other.
    """
    
    def __init__(self, method: RegistrationMethod, ratio: float = 0.75):
        """
        Initialize the registration pipeline.

        Args:
            method: The registration method to use (RegistrationMethod enum).
            ratio: Lowe's ratio test threshold used to discard ambiguous matches. Lower is stricter.
        """
        task = RegistrationTask(method=method, ratio=ratio)
        super().__init__(task=task)
        self.registrator = task  # Alias for direct task access

    def run_pipeline(self, image1: np.ndarray, image2: np.ndarray) -> RegistrationResult:
        """
        Run the registration pipeline on two images.

        Args:
            image1: The reference image.
            image2: The image to be registered to image1.

        Returns:
            RegistrationResult with:
            - registered_image: image2 warped to align with image1, keeping its original channels
            - transform: 3x3 homography mapping image2 coordinates to image1 coordinates
            - matches: (N, 4) array of inlier correspondences (x1, y1, x2, y2)
            - inlier_ratio: fraction of candidate matches kept as inliers
        """
        return super().run_pipeline(image1, image2)

    def plot_matches(self, image1: np.ndarray, image2: np.ndarray, matches: np.ndarray):
        """
        Plot the matches between two images.

        Args:
            image1: First image (as a numpy array, RGB or grayscale).
            image2: Second image (as a numpy array, RGB or grayscale).
            matches: Array of shape (N, 4), one row per match: x1, y1 in image1 and x2, y2 in image2
                (RegistrationResult.matches).
        """
        # Create a combined image by stacking the two images horizontally
        combined_image = np.hstack((image1, image2))

        # Plot the combined image
        plt.figure(figsize=(10, 5))
        if len(combined_image.shape) == 2:
            plt.imshow(combined_image, cmap='gray')
        else:
            plt.imshow(combined_image)

        # Plot each matched pair, shifting image2 coordinates by the width of image1
        for x1, y1, x2, y2 in matches:
            x2 = x2 + image1.shape[1]
            plt.plot([x1, x2], [y1, y2], color='yellow', linewidth=0.5)
            plt.scatter([x1, x2], [y1, y2], color='red', s=10)

        plt.title('Matched Keypoints Between The Two Images')
        plt.axis('off')
        plt.show()
