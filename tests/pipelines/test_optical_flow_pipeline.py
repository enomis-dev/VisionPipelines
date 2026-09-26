import pytest
import numpy as np
import cv2
from visionpipelines import OpticalFlowMethod, OpticalFlowPipeline


@pytest.fixture
def setup_images():
    """Fixture to load test images."""
    image1 = cv2.imread('tests/data/IMG1_low_res.jpg')
    image2 = cv2.imread('tests/data/IMG2_low_res.jpg')

    # Ensure images were loaded
    assert image1 is not None, "Failed to load IMG1_low_res.jpg"
    assert image2 is not None, "Failed to load IMG2_low_res.jpg"

    return image1, image2


@pytest.mark.parametrize("method", [OpticalFlowMethod.FARNEBACK, OpticalFlowMethod.RAFT_SMALL])
def test_optical_flow_pipeline(setup_images, method):
    """Test optical flow pipeline end to end on real images."""
    image1, image2 = setup_images

    pipeline = OpticalFlowPipeline(method)
    flow = pipeline.run_pipeline(image1, image2)

    assert flow.shape == (*image1.shape[:2], 2)
    assert np.isfinite(flow).all()

    assert pipeline.warp(image2, flow).shape == image2.shape
    assert pipeline.flow_to_color(flow).shape == image1.shape


def test_optical_flow_pipeline_task_access():
    """Test that the task can be accessed through the pipeline."""
    pipeline = OpticalFlowPipeline(OpticalFlowMethod.FARNEBACK)

    assert pipeline.task is not None
    assert pipeline.task == pipeline.estimator
    assert pipeline.task.method == OpticalFlowMethod.FARNEBACK
