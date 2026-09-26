import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from visionpipelines.tasks.task import Task
from visionpipelines.constants import RegistrationMethod


@dataclass
class RegistrationResult:
    """
    Result of registering a moving image onto a reference image.

    Attributes:
        registered_image: The moving image warped into the reference image's frame,
            with the same channels as the input moving image.
        transform: 3x3 homography mapping moving-image coordinates to reference-image coordinates.
        matches: (N, 4) array of RANSAC inlier correspondences, one row per match: x1, y1, x2, y2,
            where (x1, y1) is in the reference image and (x2, y2) in the moving image.
        inlier_ratio: Fraction of candidate matches kept as inliers, a rough quality indicator.
    """
    registered_image: np.ndarray
    transform: np.ndarray
    matches: np.ndarray
    inlier_ratio: float


class RegistrationTask(Task):
    """
    Task for registering (aligning) two images.

    Registration runs in three steps:
    1. Find point correspondences between the images (method specific).
    2. Estimate a homography from the correspondences with RANSAC.
    3. Warp the original moving image into the reference image's frame.
    """

    def __init__(self, method: RegistrationMethod = RegistrationMethod.ORB, ratio: float = 0.75):
        """
        Initialize the registration task with the method to use.

        :param method: The method used for keypoint detection and matching (RegistrationMethod).
        :param ratio: Lowe's ratio test threshold. A match is kept only if its distance is below
                      ``ratio`` times the distance of the second-best candidate. Lower is stricter.
        """
        if not 0 < ratio <= 1:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        self.method = method
        self.ratio = ratio

    def execute(self, image1: np.ndarray, image2: np.ndarray) -> RegistrationResult:
        """
        Register image2 to image1 using the specified method.

        Args:
            image1: The reference image (BGR or grayscale).
            image2: The image to be registered (BGR or grayscale).

        Returns:
            RegistrationResult with the warped image, the homography and the inlier matches.
        """
        points1, points2 = self._find_correspondences(self._to_grayscale(image1), self._to_grayscale(image2))
        transform, inlier_mask = self._estimate_transform(points1, points2)

        height, width = image1.shape[:2]
        registered_image = cv2.warpPerspective(image2, transform, (width, height))

        inliers = inlier_mask.ravel().astype(bool)
        matches = np.hstack((points1[inliers], points2[inliers]))

        return RegistrationResult(
            registered_image=registered_image,
            transform=transform,
            matches=matches,
            inlier_ratio=float(inliers.mean()),
        )

    def _find_correspondences(self, gray1: np.ndarray, gray2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and match keypoints between two grayscale images.

        Returns:
            Tuple of (points1, points2), each an (N, 2) float32 array of matched x, y coordinates.
        """
        if self.method == RegistrationMethod.ORB:
            detector = cv2.ORB_create()
            # ORB uses binary descriptors, so use HAMMING distance
            norm_type = cv2.NORM_HAMMING
        elif self.method == RegistrationMethod.SIFT:
            detector = cv2.SIFT_create()
            # SIFT uses float descriptors, so use L2 distance
            norm_type = cv2.NORM_L2
        else:
            raise ValueError(f"Unknown method: {self.method}")

        keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)

        # Check if descriptors were found
        if descriptors1 is None or descriptors2 is None:
            raise ValueError("No descriptors found in one or both images. Cannot perform registration.")

        if len(keypoints1) == 0 or len(keypoints2) == 0:
            raise ValueError("No keypoints found in one or both images. Cannot perform registration.")

        # Find the two nearest candidates for each descriptor, then apply Lowe's ratio test:
        # keep a match only if it is clearly better than the runner-up, discarding ambiguous ones
        bf = cv2.BFMatcher(norm_type)
        matches = [
            candidates[0]
            for candidates in bf.knnMatch(descriptors1, descriptors2, k=2)
            if len(candidates) == 2 and candidates[0].distance < self.ratio * candidates[1].distance
        ]

        # Check if we have enough matches (need at least 4 for homography)
        if len(matches) < 4:
            raise ValueError(
                f"Insufficient matches found ({len(matches)}). "
                "At least 4 matches are required for homography computation. "
                "Try using images with more distinctive features."
            )

        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
        return points1, points2

    @staticmethod
    def _estimate_transform(points1: np.ndarray, points2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the homography mapping points2 onto points1 with RANSAC.

        Returns:
            Tuple of (homography, inlier_mask).
        """
        # Note: Even with 4+ matches, homography can fail if points are in degenerate configuration
        try:
            H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
        except cv2.error as e:
            raise ValueError(
                f"Failed to compute homography: {str(e)}. "
                f"Found {len(points1)} matches, but they may be in a degenerate configuration. "
                "Try using images with more distinctive features or different viewpoints."
            ) from e

        # Check if homography was computed successfully
        if H is None:
            raise ValueError(
                f"Failed to compute homography. Found {len(points1)} matches, "
                "but RANSAC could not find a valid transformation. "
                "The images may be too different or have insufficient quality matches."
            )

        return H, mask

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert a BGR image to grayscale, leaving grayscale images unchanged."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
