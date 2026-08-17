import cv2
import torch
import numpy as np
from typing import Optional
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    FCN_ResNet50_Weights,
    deeplabv3_resnet50,
    fcn_resnet50,
)
from visionpipelines.tasks.task import Task
from visionpipelines.constants import SegmentationMethod


class SegmentationTask(Task):
    """
    Task for semantic segmentation of images.

    Assigns each pixel a class label, producing a mask the same size as the
    input image.
    """

    def __init__(
        self,
        method: SegmentationMethod = SegmentationMethod.DEEPLABV3,
        model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the segmentation task with the method and optionally the model to use.

        :param method: The method used for segmentation (SegmentationMethod).
        :param model: Optional, a pre-trained segmentation model. If not provided, a default
                      model is loaded based on the selected method.
        :param device: Device to run inference on. Defaults to CUDA if available, else CPU.
        """
        self.method = method
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights = self._default_weights(method)
        self.categories = self.weights.meta["categories"]
        self.model = model if model is not None else self.load_default_model(method)

        self.model.to(self.device)
        self.model.eval()

    def _default_weights(self, method: SegmentationMethod):
        """Return the pretrained weights enum used for the given method."""
        if method == SegmentationMethod.DEEPLABV3:
            return DeepLabV3_ResNet50_Weights.DEFAULT
        elif method == SegmentationMethod.FCN:
            return FCN_ResNet50_Weights.DEFAULT
        else:
            raise ValueError(f"Unknown method: {method}")

    def load_default_model(self, method: SegmentationMethod) -> torch.nn.Module:
        """
        Load the default model based on the selected segmentation method.

        :param method: The method used for segmentation (SegmentationMethod).
        :return: The loaded pre-trained model.
        """
        if method == SegmentationMethod.DEEPLABV3:
            return deeplabv3_resnet50(weights=self.weights)
        elif method == SegmentationMethod.FCN:
            return fcn_resnet50(weights=self.weights)
        else:
            raise ValueError(f"Unknown method: {method}")

    def execute(self, image: torch.Tensor) -> torch.Tensor:
        """
        Execute segmentation on the preprocessed image.

        Args:
            image: Preprocessed image tensor.

        Returns:
            Raw per-class score tensor of shape (1, num_classes, H, W).
        """
        if self.method not in (SegmentationMethod.DEEPLABV3, SegmentationMethod.FCN):
            raise ValueError(f"Unknown method: {self.method}")

        with torch.no_grad():
            return self.model(image)["out"]

    def pre_process(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess the image for the model input.

        Args:
            image: Input image as numpy array (BGR format from OpenCV).

        Returns:
            Preprocessed image as torch tensor.
        """
        self._original_size = image.shape[:2]

        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
        image_tensor = self.weights.transforms()(image_tensor).unsqueeze(0).to(self.device)
        return image_tensor

    def post_process(self, output: torch.Tensor) -> np.ndarray:
        """
        Post-process the output of the segmentation task.

        Args:
            output: Raw per-class score tensor of shape (1, num_classes, H, W).

        Returns:
            Class-index mask as a uint8 numpy array, resized to the original image size.
        """
        mask = output.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        height, width = self._original_size
        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    def overlay_mask(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Blend a class-index mask over the original image for visualization.

        Args:
            image: Original image the mask was computed from.
            mask: Class-index mask returned by post_process.
            alpha: Opacity of the mask overlay (0 = invisible, 1 = opaque).

        Returns:
            Image with the segmentation mask blended on top.
        """
        palette = self._color_palette(len(self.categories))
        color_mask = palette[mask]
        return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

    @staticmethod
    def _color_palette(num_classes: int) -> np.ndarray:
        """Generate a deterministic BGR color per class index (Pascal VOC style)."""
        palette = np.zeros((num_classes, 3), dtype=np.uint8)
        for class_idx in range(num_classes):
            r = g = b = 0
            c = class_idx
            for bit in range(8):
                r |= ((c >> 0) & 1) << (7 - bit)
                g |= ((c >> 1) & 1) << (7 - bit)
                b |= ((c >> 2) & 1) << (7 - bit)
                c >>= 3
            palette[class_idx] = [b, g, r]
        return palette
