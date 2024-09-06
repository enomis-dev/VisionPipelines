import cv2
import torch
import numpy as np
from typing import Union, Tuple
from visionpipelines.tasks.task import Task
from visionpipelines.constants import RegistrationMethod

class RegistrationTask(Task):
    def __init__(self, method: RegistrationMethod = RegistrationMethod.ORB):
        """
        Initialize the registration task with the reference image and the method to use.

        :param image1: The reference image to which other images will be registered.
        :param method: The method used for keypoint detection and matching (RegistrationMethod).
        """
        self.method = method

    def register_images(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

        # Compute Homography with RANSAC
        H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

        # Use only inliers (matches deemed correct by RANSAC)
        inliers_points1 = points1[mask.ravel() == 1]
        inliers_points2 = points2[mask.ravel() == 1]

        # Warp image2 to align with image1
        height, width = image1.shape[:2]
        registered_image = cv2.warpPerspective(image2, H, (width, height))

        # Prepare the keypoints output array (4xN), using only the inliers
        keypoints = np.vstack((inliers_points1.squeeze(1).T, inliers_points2.squeeze(1).T))

        return registered_image, keypoints

    def pre_process(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        # Convert images to grayscale if they are in RGB
        if len(image1.shape) == 3:
            image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            image1_gray = image1

        if len(image2.shape) == 3:
            image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            image2_gray = image2

        return image1_gray, image2_gray
