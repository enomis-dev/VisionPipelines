import torch
import numpy as np
from typing import Optional
from visionpipelines.tasks.segmentation_task import SegmentationTask
from visionpipelines.constants import SegmentationMethod
from visionpipelines.pipelines.vision_pipeline import TaskBasedPipeline


class SegmentationPipeline(TaskBasedPipeline):
    """
    Pipeline for semantic segmentation of images.

    This pipeline assigns each pixel of an image a class label using a
    segmentation model (e.g. DeepLabV3) and returns a class-index mask.
    """

    def __init__(
        self,
        method: SegmentationMethod,
        model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the segmentation pipeline.

        Args:
            method: The segmentation method to use (SegmentationMethod enum).
            model: Optional pre-trained model. If None, a default model is loaded.
            device: Device to run inference on. Defaults to CUDA if available, else CPU.
        """
        task = SegmentationTask(method=method, model=model, device=device)
        super().__init__(task=task)
        self.segmenter = task  # Alias for direct task access

    def run_pipeline(self, image: np.ndarray) -> np.ndarray:
        """
        Run the segmentation pipeline on an image.

        Args:
            image: Input image as numpy array.

        Returns:
            Class-index mask as a numpy array the same height/width as the input image.
        """
        return super().run_pipeline(image)

    def overlay_mask(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Blend a segmentation mask over the original image for visualization.

        Args:
            image: Original image the mask was computed from.
            mask: Class-index mask returned by run_pipeline.
            alpha: Opacity of the mask overlay (0 = invisible, 1 = opaque).

        Returns:
            Image with the segmentation mask blended on top.
        """
        return self.segmenter.overlay_mask(image, mask, alpha)
