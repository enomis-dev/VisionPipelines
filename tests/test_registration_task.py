import pytest
import numpy as np
import cv2
import torch
from src.registration_task import RegistrationTask
from src.constants import RegistrationMethod
from PIL import Image


@pytest.fixture
def setup_images():
    # Load images using OpenCV
    image1 = cv2.imread('tests/data/IMG1_low_res.jpg')
    image2 = cv2.imread('tests/data/IMG1_low_res.jpg')

    return image1, image2

def test_orb_registration(setup_images):
    image1, image2 = setup_images

    # Test using ORB method
    task = RegistrationTask(image1, method=RegistrationMethod.ORB)
    registered_image, keypoints = task(image2)

    # Check if the registered image has the same shape as the input images
    assert registered_image.shape == image1.shape

    # Check the shape of the keypoints array
    assert keypoints.shape[0] == 4
    assert keypoints.shape[1] > 0  # Ensure keypoints were found

    # Check that keypoints for both images were found
    assert keypoints[0].size == keypoints[1].size  # Image1 keypoints
    assert keypoints[2].size == keypoints[3].size  # Image2 keypoints
