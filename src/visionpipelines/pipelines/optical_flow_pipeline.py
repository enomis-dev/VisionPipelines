import torch
import numpy as np
from typing import Optional
from visionpipelines.tasks.optical_flow_task import OpticalFlowTask
from visionpipelines.constants import OpticalFlowMethod
from visionpipelines.pipelines.vision_pipeline import TaskBasedPipeline


class OpticalFlowPipeline(TaskBasedPipeline):
    """
    Pipeline for dense optical flow between two images.

    This pipeline estimates, for every pixel of the first image, where it moved to in the
    second image, using either classical Farneback or the deep-learning RAFT models.
    """

    def __init__(
        self,
        method: OpticalFlowMethod,
        model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the optical flow pipeline.

        Args:
            method: The optical flow method to use (OpticalFlowMethod enum).
            model: Optional pre-trained RAFT model. If None, a default model is loaded.
            device: Device to run inference on. Defaults to CUDA if available, else CPU.
        """
        task = OpticalFlowTask(method=method, model=model, device=device)
        super().__init__(task=task)
        self.estimator = task  # Alias for direct task access

    def run_pipeline(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        Run the optical flow pipeline on two images.

        Args:
            image1: The first image.
            image2: The second image, same height and width as image1.

        Returns:
            Flow field as a float32 array of shape (H, W, 2) holding (dx, dy) per pixel of image1.
        """
        return super().run_pipeline(image1, image2)

    def warp(self, image2: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        Warp image2 into image1's frame using the flow (dense, non-rigid registration).

        Args:
            image2: The second image the flow was computed towards.
            flow: Flow field returned by run_pipeline.

        Returns:
            image2 resampled to line up with image1.
        """
        return self.estimator.warp(image2, flow)

    def flow_to_color(self, flow: np.ndarray) -> np.ndarray:
        """
        Visualize a flow field as an RGB image (hue = direction, saturation = magnitude).

        Args:
            flow: Flow field returned by run_pipeline.

        Returns:
            RGB uint8 image of shape (H, W, 3).
        """
        return self.estimator.flow_to_color(flow)
