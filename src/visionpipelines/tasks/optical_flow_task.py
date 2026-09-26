import cv2
import torch
import numpy as np
from typing import Optional
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small
from torchvision.utils import flow_to_image
from visionpipelines.tasks.task import Task
from visionpipelines.constants import OpticalFlowMethod

# RAFT downsamples by 8, so its inputs must have height and width divisible by 8,
# and its correlation pyramid needs feature maps of at least 16 cells, i.e. 128 pixels
RAFT_SIZE_MULTIPLE = 8
RAFT_MIN_SIZE = 128


class OpticalFlowTask(Task):
    """
    Task for dense optical flow between two images.

    Computes a per-pixel displacement field: for every pixel (x, y) of image1, flow[y, x] = (dx, dy)
    is where that pixel moved to in image2, i.e. image1(x, y) ~ image2(x + dx, y + dy).
    """

    def __init__(
        self,
        method: OpticalFlowMethod = OpticalFlowMethod.RAFT_LARGE,
        model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the optical flow task with the method and optionally the model to use.

        :param method: The method used to compute optical flow (OpticalFlowMethod).
        :param model: Optional, a pre-trained RAFT model. If not provided, a default model is loaded
                      based on the selected method. Ignored for FARNEBACK, which has no model.
        :param device: Device to run inference on. Defaults to CUDA if available, else CPU.
        """
        self.method = method
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights = None
        self.model = None

        if method in (OpticalFlowMethod.RAFT_SMALL, OpticalFlowMethod.RAFT_LARGE):
            self.weights = self._default_weights(method)
            self.model = model if model is not None else self.load_default_model(method)
            self.model.to(self.device)
            self.model.eval()
        elif method != OpticalFlowMethod.FARNEBACK:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def _default_weights(method: OpticalFlowMethod):
        """Return the pretrained weights enum used for the given RAFT method."""
        if method == OpticalFlowMethod.RAFT_SMALL:
            return Raft_Small_Weights.DEFAULT
        elif method == OpticalFlowMethod.RAFT_LARGE:
            return Raft_Large_Weights.DEFAULT
        else:
            raise ValueError(f"Unknown method: {method}")

    def load_default_model(self, method: OpticalFlowMethod) -> torch.nn.Module:
        """
        Load the default model based on the selected optical flow method.

        :param method: The method used for optical flow (OpticalFlowMethod).
        :return: The loaded pre-trained model.
        """
        if method == OpticalFlowMethod.RAFT_SMALL:
            return raft_small(weights=self.weights)
        elif method == OpticalFlowMethod.RAFT_LARGE:
            return raft_large(weights=self.weights)
        else:
            raise ValueError(f"Unknown method: {method}")

    def execute(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        Compute the optical flow from image1 to image2.

        Args:
            image1: The first image (BGR or grayscale).
            image2: The second image (BGR or grayscale), same height and width as image1.

        Returns:
            Flow field as a float32 array of shape (H, W, 2) holding (dx, dy) per pixel of image1.
        """
        if image1.shape[:2] != image2.shape[:2]:
            raise ValueError(
                f"Images must have the same height and width, got {image1.shape[:2]} and {image2.shape[:2]}"
            )

        if self.method == OpticalFlowMethod.FARNEBACK:
            return self._flow_farneback(image1, image2)
        elif self.method in (OpticalFlowMethod.RAFT_SMALL, OpticalFlowMethod.RAFT_LARGE):
            return self._flow_raft(image1, image2)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    @staticmethod
    def _flow_farneback(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """Compute dense flow with OpenCV's classical Farneback algorithm."""
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if image1.ndim == 3 else image1
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if image2.ndim == 3 else image2
        return cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

    def _flow_raft(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """Compute dense flow with a pretrained RAFT model."""
        height, width = image1.shape[:2]
        batch1, batch2 = self.weights.transforms()(self._to_rgb_tensor(image1), self._to_rgb_tensor(image2))

        # Pad bottom/right to a valid RAFT size and crop the flow back afterwards, so no rescaling is needed
        pad_bottom = self._raft_padded_size(height) - height
        pad_right = self._raft_padded_size(width) - width
        padding = (0, pad_right, 0, pad_bottom)
        batch1 = torch.nn.functional.pad(batch1, padding, mode='replicate').to(self.device)
        batch2 = torch.nn.functional.pad(batch2, padding, mode='replicate').to(self.device)

        with torch.no_grad():
            # RAFT refines the flow iteratively and returns every iteration; the last one is the most accurate
            flow = self.model(batch1, batch2)[-1]

        return flow[0, :, :height, :width].permute(1, 2, 0).cpu().numpy()

    @staticmethod
    def _raft_padded_size(size: int) -> int:
        """Smallest size >= the input that RAFT accepts: a multiple of 8 and at least 128."""
        return max(RAFT_MIN_SIZE, size + (-size % RAFT_SIZE_MULTIPLE))

    @staticmethod
    def _to_rgb_tensor(image: np.ndarray) -> torch.Tensor:
        """Convert a BGR or grayscale uint8 image to a (1, 3, H, W) RGB uint8 tensor."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB if image.ndim == 3 else cv2.COLOR_GRAY2RGB)
        return torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def warp(image2: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        Warp image2 into image1's frame using the flow, i.e. dense (non-rigid) registration.

        Args:
            image2: The second image the flow was computed towards.
            flow: Flow field of shape (H, W, 2) from image1 to image2.

        Returns:
            image2 resampled so that each pixel lines up with the corresponding pixel of image1.
        """
        height, width = flow.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        return cv2.remap(image2, grid_x + flow[..., 0], grid_y + flow[..., 1], cv2.INTER_LINEAR)

    @staticmethod
    def flow_to_color(flow: np.ndarray) -> np.ndarray:
        """
        Visualize a flow field as a color image: hue encodes direction, saturation encodes magnitude.

        Args:
            flow: Flow field of shape (H, W, 2).

        Returns:
            RGB uint8 image of shape (H, W, 3).
        """
        flow_tensor = torch.from_numpy(np.ascontiguousarray(flow)).permute(2, 0, 1).float()
        return flow_to_image(flow_tensor).permute(1, 2, 0).numpy()
