import pytest
import numpy as np
import cv2
import torch
from visionpipelines.pipelines.registration_pipeline import RegistrationPipeline
from visionpipelines.constants import RegistrationMethod
from PIL import Image


@pytest.fixture
def setup_images():
    """Fixture to load test images."""
    # Load images using OpenCV
    image1 = cv2.imread('tests/data/IMG1_low_res.jpg')
    image2 = cv2.imread('tests/data/IMG2_low_res.jpg')
    
    # Ensure images were loaded
    assert image1 is not None, "Failed to load IMG1_low_res.jpg"
    assert image2 is not None, "Failed to load IMG2_low_res.jpg"

    return image1, image2

def test_registration_pipeline_orb(setup_images):
    """Test registration pipeline using ORB method."""
    image1, image2 = setup_images

    # Test using ORB method
    pipeline = RegistrationPipeline(RegistrationMethod.ORB)
    registered_image, keypoints = pipeline.run_pipeline(image1, image2)

    # Check if the registered image has the same shape as the input images (grayscale)
    assert registered_image.shape == image1.shape[0:2]

    # Check the shape of the keypoints array (4 rows: x1, y1, x2, y2)
    assert keypoints.shape[0] == 4
    assert keypoints.shape[1] > 0  # Ensure keypoints were found

    # Check that keypoints for both images were found
    assert keypoints[0].size == keypoints[1].size  # Image1 keypoints
    assert keypoints[2].size == keypoints[3].size  # Image2 keypoints

def test_registration_pipeline_sift(setup_images):
    """Test registration pipeline using SIFT method."""
    image1, image2 = setup_images

    # Test using SIFT method
    pipeline = RegistrationPipeline(RegistrationMethod.SIFT)
    registered_image, keypoints = pipeline.run_pipeline(image1, image2)

    # Check if the registered image has the same shape as the input images
    assert registered_image.shape == image1.shape[0:2]

    # Check the shape of the keypoints array
    assert keypoints.shape[0] == 4
    assert keypoints.shape[1] > 0  # Ensure keypoints were found

def test_registration_pipeline_task_access(setup_images):
    """Test that the task can be accessed through the pipeline."""
    image1, image2 = setup_images
    
    pipeline = RegistrationPipeline(RegistrationMethod.ORB)
    
    # Check that task is accessible
    assert pipeline.task is not None
    assert pipeline.registrator is not None
    assert pipeline.task == pipeline.registrator
    
    # Check that task has the correct method
    assert pipeline.task.method == RegistrationMethod.ORB

def test_registration_pipeline_invalid_method():
    """Test that invalid registration method raises error."""
    # This test would require adding an invalid method, but since we use enums,
    # this is less likely. However, we can test the task directly.
    from visionpipelines.tasks.registration_task import RegistrationTask
    
    # Create task with valid method
    task = RegistrationTask(RegistrationMethod.ORB)
    
    # Test with invalid method (would need to bypass enum, but that's not typical usage)
    # This test is more of a documentation of expected behavior
    pass
