import pytest
import numpy as np
import torch
from visionpipelines.tasks.segmentation_task import SegmentationTask
from visionpipelines.constants import SegmentationMethod


class TestSegmentationTask:
    """Test suite for SegmentationTask."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    def test_task_initialization_deeplabv3(self):
        """Test task initialization with DeepLabV3 method."""
        task = SegmentationTask(SegmentationMethod.DEEPLABV3)
        assert task.method == SegmentationMethod.DEEPLABV3
        assert task.model is not None
        assert task.device is not None
        assert "__background__" in task.categories

    def test_task_initialization_fcn(self):
        """Test task initialization with FCN method."""
        task = SegmentationTask(SegmentationMethod.FCN)
        assert task.method == SegmentationMethod.FCN
        assert task.model is not None

    def test_pre_process_bgr_to_rgb(self, sample_image):
        """Test preprocessing converts BGR to RGB and creates a batched tensor."""
        task = SegmentationTask(SegmentationMethod.DEEPLABV3)

        tensor = task.pre_process(sample_image)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dim() == 4  # Batch dimension added
        assert tensor.shape[0] == 1
        assert tensor.shape[1] == 3  # RGB channels

    def test_pre_process_grayscale(self):
        """Test preprocessing with grayscale image."""
        task = SegmentationTask(SegmentationMethod.DEEPLABV3)

        gray_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        tensor = task.pre_process(gray_image)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dim() == 4

    def test_execute_returns_class_scores(self, sample_image):
        """Test execute method returns per-class scores."""
        task = SegmentationTask(SegmentationMethod.DEEPLABV3)

        tensor = task.pre_process(sample_image)
        output = task.execute(tensor)

        assert isinstance(output, torch.Tensor)
        assert output.dim() == 4  # (batch, num_classes, H, W)
        assert output.shape[1] == len(task.categories)

    def test_post_process_mask_matches_input_size(self, sample_image):
        """Test post_process resizes the mask back to the original image size."""
        task = SegmentationTask(SegmentationMethod.DEEPLABV3)

        tensor = task.pre_process(sample_image)
        output = task.execute(tensor)
        mask = task.post_process(output)

        assert isinstance(mask, np.ndarray)
        assert mask.shape == sample_image.shape[:2]
        assert mask.dtype == np.uint8
        assert mask.max() < len(task.categories)

    def test_overlay_mask(self, sample_image):
        """Test overlaying the mask on the original image."""
        task = SegmentationTask(SegmentationMethod.DEEPLABV3)

        tensor = task.pre_process(sample_image)
        output = task.execute(tensor)
        mask = task.post_process(output)

        overlaid = task.overlay_mask(sample_image, mask)

        assert isinstance(overlaid, np.ndarray)
        assert overlaid.shape == sample_image.shape
        assert overlaid.dtype == sample_image.dtype

    def test_invalid_method_raises_error(self):
        """Test that an invalid method raises an error in execute."""
        task = SegmentationTask(SegmentationMethod.DEEPLABV3)
        task.method = "INVALID"

        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        tensor = task.pre_process(image)

        with pytest.raises(ValueError, match="Unknown method"):
            task.execute(tensor)
