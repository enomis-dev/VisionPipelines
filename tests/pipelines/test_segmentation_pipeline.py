import pytest
import numpy as np
import cv2
from visionpipelines.pipelines.segmentation_pipeline import SegmentationPipeline
from visionpipelines.constants import SegmentationMethod


class TestSegmentationPipeline:
    """Test suite for SegmentationPipeline."""

    @pytest.fixture
    def sample_image(self):
        """Load a sample image for testing."""
        image_path = 'tests/data/people_image.webp'
        image = cv2.imread(image_path)

        if image is None:
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        return image

    def test_pipeline_initialization(self):
        """Test pipeline initialization with DeepLabV3."""
        pipeline = SegmentationPipeline(SegmentationMethod.DEEPLABV3)

        assert pipeline.task is not None
        assert pipeline.segmenter is not None
        assert pipeline.task == pipeline.segmenter
        assert pipeline.task.method == SegmentationMethod.DEEPLABV3

    def test_pipeline_run_pipeline(self, sample_image):
        """Test running the segmentation pipeline."""
        pipeline = SegmentationPipeline(SegmentationMethod.DEEPLABV3)

        mask = pipeline.run_pipeline(sample_image)

        assert isinstance(mask, np.ndarray)
        assert mask.shape == sample_image.shape[:2]

    def test_pipeline_overlay_mask(self, sample_image):
        """Test overlaying the mask on the original image."""
        pipeline = SegmentationPipeline(SegmentationMethod.DEEPLABV3)

        mask = pipeline.run_pipeline(sample_image)
        overlaid = pipeline.overlay_mask(sample_image, mask)

        assert isinstance(overlaid, np.ndarray)
        assert overlaid.shape == sample_image.shape
        assert overlaid.dtype == sample_image.dtype
