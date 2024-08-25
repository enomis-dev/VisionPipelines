import pytest
import numpy as np
import cv2
import torch
from src.registration_task import RegistrationTask
from src.constants import RegistrationMethod

@pytest.fixture
def setup_images():
    # Create a simple synthetic image for testing
    image1 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(image1, (50, 50), 10, (255, 255, 255), -1)  # Draw a white circle in the center

    # Create a second image by translating the first image
    image2 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(image2, (60, 60), 10, (255, 255, 255), -1)  # Draw the same circle but shifted

    # Convert the second image to a tensor format for the pipeline
    image2_tensor = torch.tensor(image2).float() / 255.0
    image2_tensor = image2_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC to NCHW

    return image1, image2_tensor

def test_orb_registration(setup_images):
    image1, image2_tensor = setup_images

    # Instantiate the RegistrationTask with ORB method
    registration_task = RegistrationTask(image1=image1, method=RegistrationMethod.ORB)

    # Perform the registration
    registered_image_tensor = registration_task(image2_tensor)

    # Convert the tensor back to numpy for verification
    registered_image = (registered_image_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Check if the registered image has the circle correctly aligned with the reference image
    difference = cv2.absdiff(image1, registered_image)
    assert np.sum(difference) < 1e-3, "The registration did not align the images correctly using ORB."

def test_sift_registration(setup_images):
    image1, image2_tensor = setup_images

    # Instantiate the RegistrationTask with SIFT method
    registration_task = RegistrationTask(image1=image1, method=RegistrationMethod.SIFT)

    # Perform the registration
    registered_image_tensor = registration_task(image2_tensor)

    # Convert the tensor back to numpy for verification
    registered_image = (registered_image_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Check if the registered image has the circle correctly aligned with the reference image
    difference = cv2.absdiff(image1, registered_image)
    assert np.sum(difference) < 1e-3, "The registration did not align the images correctly using SIFT."
