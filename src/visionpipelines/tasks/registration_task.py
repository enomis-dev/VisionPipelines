import cv2
import torch
import numpy as np
from typing import Union, Tuple
from visionpipelines.tasks.task import Task
from visionpipelines.constants import RegistrationMethod

class RegistrationTask(Task):
    """
    Task for registering (aligning) two images.
    
    This task detects keypoints in both images, matches them, and computes
    a transformation to align one image with the other.
    """
    
    def __init__(self, method: RegistrationMethod = RegistrationMethod.ORB):
        """
        Initialize the registration task with the method to use.

        :param method: The method used for keypoint detection and matching (RegistrationMethod).
        """
        self.method = method

    def execute(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Register image2 to image1 using the specified method.
        
        Args:
            image1: The reference image (grayscale).
            image2: The image to be registered (grayscale).
            
        Returns:
            Tuple of (registered_image, keypoints) where:
            - registered_image: image2 warped to align with image1
            - keypoints: Array of shape (4, N) containing matched keypoint coordinates
        """
        if self.method == RegistrationMethod.ORB:
            # ORB Detector
            orb = cv2.ORB_create()
            keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
            keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
            # ORB uses binary descriptors, so use HAMMING distance
            norm_type = cv2.NORM_HAMMING
        elif self.method == RegistrationMethod.SIFT:
            # SIFT Detector
            sift = cv2.SIFT_create()
            keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
            keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
            # SIFT uses float descriptors, so use L2 distance
            norm_type = cv2.NORM_L2
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Check if descriptors were found
        if descriptors1 is None or descriptors2 is None:
            raise ValueError("No descriptors found in one or both images. Cannot perform registration.")
        
        if len(keypoints1) == 0 or len(keypoints2) == 0:
            raise ValueError("No keypoints found in one or both images. Cannot perform registration.")

        # Match descriptors using BFMatcher with appropriate norm type
        bf = cv2.BFMatcher(norm_type, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Check if we have enough matches (need at least 4 for homography)
        if len(matches) < 4:
            raise ValueError(
                f"Insufficient matches found ({len(matches)}). "
                "At least 4 matches are required for homography computation. "
                "Try using images with more distinctive features."
            )

        # Extract location of good matches
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute Homography with RANSAC
        # Note: Even with 4+ matches, homography can fail if points are in degenerate configuration
        try:
            H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
        except cv2.error as e:
            raise ValueError(
                f"Failed to compute homography: {str(e)}. "
                f"Found {len(matches)} matches, but they may be in a degenerate configuration. "
                "Try using images with more distinctive features or different viewpoints."
            ) from e

        # Check if homography was computed successfully
        if H is None:
            raise ValueError(
                f"Failed to compute homography. Found {len(matches)} matches, "
                "but RANSAC could not find a valid transformation. "
                "The images may be too different or have insufficient quality matches."
            )

        # Use only inliers (matches deemed correct by RANSAC)
        inliers_points1 = points1[mask.ravel() == 1]
        inliers_points2 = points2[mask.ravel() == 1]

        # Warp image2 to align with image1
        height, width = image1.shape[:2]
        registered_image = cv2.warpPerspective(image2, H, (width, height))

        # Prepare the keypoints output array (4xN), using only the inliers
        if len(inliers_points1) > 0:
            keypoints = np.vstack((inliers_points1.squeeze(1).T, inliers_points2.squeeze(1).T))
        else:
            # Fallback: use all matches if no inliers
            keypoints = np.vstack((points1.squeeze(1).T, points2.squeeze(1).T))

        return registered_image, keypoints

    def pre_process(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess images by converting to grayscale if needed.
        
        Args:
            image1: First input image (BGR or grayscale).
            image2: Second input image (BGR or grayscale).
            
        Returns:
            Tuple of (image1_gray, image2_gray) as grayscale images.
        """
        # Convert images to grayscale if they are in RGB/BGR
        if len(image1.shape) == 3:
            image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            image1_gray = image1

        if len(image2.shape) == 3:
            image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            image2_gray = image2

        return image1_gray, image2_gray
    
    def post_process(self, registered_image: np.ndarray, keypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Postprocess registration results (currently just passes through).
        
        Args:
            registered_image: The registered image.
            keypoints: The matched keypoints.
            
        Returns:
            Tuple of (registered_image, keypoints).
        """
        return registered_image, keypoints
