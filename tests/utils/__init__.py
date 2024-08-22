import cv2
import torch
import numpy as np
import pytest
from vision_pipeline import VisionPipeline
from vision_pipeline.utils import resize, to_grayscale, apply_threshold

@pytest.fixture
def sample_image():
    """Fixture to create a simple 3x3 color image."""
    return np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                     [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
                     [[255, 255, 255], [128, 128, 128], [0, 0, 0]]], dtype=np.uint8)

@pytest.fixture
def pipeline():
    """Fixture to create a VisionPipeline instance."""
    return VisionPipeline()

def test_resize(pipeline, sample_image):
    pipeline.add_task(resize((2, 2)))
    result = pipeline.run_pipeline(sample_image)

    # Check that the output is the correct size
    assert result.shape == (2, 2, 3)

def test_grayscale(pipeline, sample_image):
    pipeline.add_task(to_grayscale())
    result = pipeline.run_pipeline(sample_image)

    # Check that the output is single channel (grayscale)
    assert result.shape == (3, 3)  # Now the image should have one channel

def test_threshold(pipeline, sample_image):
    pipeline.add_task(to_grayscale())
    pipeline.add_task(apply_threshold(127))
    result = pipeline.run_pipeline(sample_image)

    # Check that the result contains only 0 and 255 (binary image)
    unique_values = np.unique(result)
    assert np.array_equal(unique_values, np.array([0, 255])) or np.array_equal(unique_values, np.array([255]))

def test_pipeline_clear(pipeline):
    pipeline.add_task(resize((2, 2)))
    pipeline.add_task(to_grayscale())

    # Clear the pipeline
    pipeline.clear_pipeline()

    # Check that the pipeline is empty
    assert len(pipeline.tasks) == 0

def test_combined_pipeline(pipeline, sample_image):
    pipeline.add_task(resize((2, 2)))
    pipeline.add_task(to_grayscale())
    pipeline.add_task(apply_threshold(127))

    result = pipeline.run_pipeline(sample_image)

    # Check final output size and type
    assert result.shape == (2, 2)  # Grayscale image resized to 2x2
    unique_values = np.unique(result)
    assert np.array_equal(unique_values, np.array([0, 255])) or np.array_equal(unique_values, np.array([255]))
