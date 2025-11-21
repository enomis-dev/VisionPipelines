import cv2
import pytest
import torch
import kornia
import numpy as np
from typing import List, Callable, Union
from torchvision.transforms import functional as F
from visionpipelines.pipelines.vision_pipeline import FunctionBasedPipeline
from visionpipelines.utils import resize, to_grayscale

@pytest.fixture
def sample_image():
    """Fixture to create a simple 3x3 color image."""
    return np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                     [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
                     [[255, 255, 255], [128, 128, 128], [0, 0, 0]]], dtype=np.uint8)

@pytest.fixture
def pipeline():
    """Fixture to create a FunctionBasedPipeline instance."""
    return FunctionBasedPipeline()

def pre_process(image: np.ndarray) -> torch.Tensor:
    """Preprocess the image for the model input."""
    # Convert image to RGB if it's in BGR (as OpenCV loads images in BGR format)
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    image_tensor = F.to_tensor(image)
    return image_tensor

def test_resize(pipeline, sample_image):
    """Test that resize function works in the pipeline."""
    pipeline.add_task(pre_process)
    pipeline.add_task(resize((2, 2)))
    result = pipeline.run_pipeline(sample_image)

    # Check that the output is the correct size
    assert result.shape == (3, 2, 2)

def test_grayscale(pipeline, sample_image):
    """Test that grayscale conversion works in the pipeline."""
    pipeline.add_task(pre_process)
    pipeline.add_task(to_grayscale())
    result = pipeline.run_pipeline(sample_image)

    # Check that the output is single channel (grayscale)
    assert result.shape == (1, 3, 3)  # Now the image should have one channel

def test_pipeline_clear(pipeline):
    """Test that clearing the pipeline removes all tasks."""
    pipeline.add_task(pre_process)
    pipeline.add_task(resize((2, 2)))
    pipeline.add_task(to_grayscale())

    # Clear the pipeline
    pipeline.clear_pipeline()

    # Check that the pipeline is empty
    assert len(pipeline.tasks) == 0

def test_pipeline_input_validation(pipeline):
    """Test that pipeline validates input correctly."""
    # Test None input
    with pytest.raises(ValueError, match="cannot be None"):
        pipeline.run_pipeline(None)
    
    # Test empty array
    with pytest.raises(ValueError, match="cannot be empty"):
        pipeline.run_pipeline(np.array([]))
    
    # Test invalid type
    with pytest.raises(ValueError, match="must be numpy array"):
        pipeline.run_pipeline("not an image")

def test_pipeline_add_invalid_task(pipeline):
    """Test that adding non-callable task raises error."""
    with pytest.raises(TypeError, match="must be callable"):
        pipeline.add_task("not a callable")
