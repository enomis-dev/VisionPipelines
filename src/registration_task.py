import cv2
import torch
import numpy as np
from typing import Union, Tuple
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

    def __call__(self, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform image registration.

        :param image2: The image to register, as a numpy array (RGB).
        :return: A tuple of the registered image and the keypoints found in each image.
                 The keypoints are returned as a numpy array of shape (4, N), where N is the number of points found.
        """
        # Perform registration
        registered_image, keypoints = self.register_images(self.image1, image2)

        return registered_image, keypoints

    def register_images(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Register image2 to image1 using the specified method."""
        # Convert images to grayscale if they are in RGB
        if len(image1.shape) == 3:
            image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            image1_gray = image1

        if len(image2.shape) == 3:
            image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            image2_gray = image2

        if self.method == RegistrationMethod.ORB:
            # ORB Detector
            orb = cv2.ORB_create()
            keypoints1, descriptors1 = orb.detectAndCompute(image1_gray, None)
            keypoints2, descriptors2 = orb.detectAndCompute(image2_gray, None)
        elif self.method == RegistrationMethod.SIFT:
            # SIFT Detector
            sift = cv2.SIFT_create()
            keypoints1, descriptors1 = sift.detectAndCompute(image1_gray, None)
            keypoints2, descriptors2 = sift.detectAndCompute(image2_gray, None)
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

        # Prepare the keypoints output array (4xN)
        keypoints = np.vstack((points1.squeeze(1).T, points2.squeeze(1).T))

        return registered_image, keypoints
