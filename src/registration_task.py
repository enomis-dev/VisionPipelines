import cv2
import torch
import numpy as np
from typing import Union
from src.task import Task
from src.constants import RegistrationMethod  # Import the RegistrationMethod enum

class RegistrationTask(Task):
    def __init__(self, image1: np.ndarray, method: RegistrationMethod = RegistrationMethod.ORB):
        """
        Initialize the registration task with the reference image and the method to use.

        :param image1: The reference image to which other images will be registered.
        :param method: The method used for keypoint detection and matching (RegistrationMethod).
        """
        self.image1 = image1
        self.method = method

    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform image registration.

        :param image_tensor: The image to register, as a PyTorch tensor.
        :return: The registered image, as a PyTorch tensor.
        """
        # Convert tensors to numpy arrays
        image2 = (image_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Perform registration
        registered_image = self.register_images(self.image1, image2)

        # Convert the registered image back to tensor
        registered_image_tensor = torch.tensor(registered_image).float() / 255.0
        registered_image_tensor = registered_image_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC to NCHW

        return registered_image_tensor

    def register_images(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """Register image2 to image1 using the specified method."""
        if self.method == RegistrationMethod.ORB:
            # ORB Detector
            orb = cv2.ORB_create()
            keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
            keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
        elif self.method == RegistrationMethod.SIFT:
            # SIFT Detector
            sift = cv2.SIFT_create()
            keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
            keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Match descriptors using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract location of good matches
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute Homography
        H, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

        # Warp image2 to align with image1
        height, width = image1.shape[:2]
        registered_image = cv2.warpPerspective(image2, H, (width, height))

        return registered_image
