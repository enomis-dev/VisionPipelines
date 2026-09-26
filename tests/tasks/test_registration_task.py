import pytest
import numpy as np
import cv2
from visionpipelines.tasks.registration_task import RegistrationResult, RegistrationTask
from visionpipelines.constants import RegistrationMethod


class TestRegistrationTask:
    """Test suite for RegistrationTask."""

    @pytest.fixture
    def shifted_images(self):
        """A real BGR image and a copy translated by a known offset."""
        image1 = cv2.imread('tests/data/IMG1_low_res.jpg')
        assert image1 is not None, "Failed to load IMG1_low_res.jpg"

        shift = np.float32([[1, 0, 15], [0, 1, 10]])  # move 15px right, 10px down
        height, width = image1.shape[:2]
        image2 = cv2.warpAffine(image1, shift, (width, height))
        return image1, image2

    @pytest.fixture
    def simple_images(self):
        """Create simple images that may not have enough features."""
        # Create two simple test images with minimal pattern
        image1 = np.zeros((100, 100), dtype=np.uint8)
        image1[30:70, 30:70] = 255  # White square

        image2 = np.zeros((100, 100), dtype=np.uint8)
        image2[35:75, 35:75] = 255  # Slightly shifted white square

        return image1, image2

    def test_task_initialization_orb(self):
        """Test task initialization with ORB method."""
        task = RegistrationTask(RegistrationMethod.ORB)
        assert task.method == RegistrationMethod.ORB

    def test_task_initialization_sift(self):
        """Test task initialization with SIFT method."""
        task = RegistrationTask(RegistrationMethod.SIFT)
        assert task.method == RegistrationMethod.SIFT

    @pytest.mark.parametrize("method", [RegistrationMethod.ORB, RegistrationMethod.SIFT])
    def test_execute_returns_result(self, shifted_images, method):
        """Test execute returns a RegistrationResult with consistent fields."""
        image1, image2 = shifted_images

        result = RegistrationTask(method).execute(image1, image2)

        assert isinstance(result, RegistrationResult)
        assert result.transform.shape == (3, 3)
        assert result.matches.ndim == 2
        assert result.matches.shape[1] == 4  # columns: x1, y1, x2, y2
        assert result.matches.shape[0] >= 4
        assert 0.0 < result.inlier_ratio <= 1.0

    @pytest.mark.parametrize("method", [RegistrationMethod.ORB, RegistrationMethod.SIFT])
    def test_execute_recovers_known_shift(self, shifted_images, method):
        """The homography should undo the known translation (-15px, -10px)."""
        image1, image2 = shifted_images

        result = RegistrationTask(method).execute(image1, image2)
        H = result.transform / result.transform[2, 2]

        assert H[0, 2] == pytest.approx(-15, abs=1.0)
        assert H[1, 2] == pytest.approx(-10, abs=1.0)
        assert np.allclose(H[:2, :2], np.eye(2), atol=0.02)

    def test_registered_image_keeps_color(self, shifted_images):
        """The warp is applied to the original image, not the grayscale copy used for matching."""
        image1, image2 = shifted_images

        result = RegistrationTask(RegistrationMethod.ORB).execute(image1, image2)

        assert result.registered_image.shape == image1.shape
        assert result.registered_image.ndim == 3

    def test_execute_grayscale_input(self, shifted_images):
        """Grayscale inputs produce a grayscale registered image."""
        image1, image2 = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in shifted_images)

        result = RegistrationTask(RegistrationMethod.ORB).execute(image1, image2)

        assert result.registered_image.shape == image1.shape

    def test_matches_are_consistent_with_transform(self, shifted_images):
        """Inlier matches in image2 should map onto their image1 counterparts through the transform."""
        image1, image2 = shifted_images

        result = RegistrationTask(RegistrationMethod.SIFT).execute(image1, image2)
        points2 = result.matches[:, 2:4].reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(points2, result.transform).reshape(-1, 2)

        # RANSAC reprojection threshold is 5px
        assert np.all(np.linalg.norm(projected - result.matches[:, 0:2], axis=1) <= 5.0)

    @pytest.mark.parametrize("ratio", [0, -0.5, 1.5])
    def test_invalid_ratio_raises_error(self, ratio):
        """The ratio test threshold must be in (0, 1]."""
        with pytest.raises(ValueError, match="ratio"):
            RegistrationTask(RegistrationMethod.ORB, ratio=ratio)

    def test_stricter_ratio_keeps_fewer_matches(self, shifted_images):
        """A stricter ratio test should never produce more matches than a looser one."""
        image1, image2 = shifted_images

        loose = RegistrationTask(RegistrationMethod.SIFT, ratio=0.9)._find_correspondences(
            *(RegistrationTask._to_grayscale(image) for image in (image1, image2)))
        strict = RegistrationTask(RegistrationMethod.SIFT, ratio=0.5)._find_correspondences(
            *(RegistrationTask._to_grayscale(image) for image in (image1, image2)))

        assert len(strict[0]) <= len(loose[0])

    def test_execute_insufficient_matches(self, simple_images):
        """Test that execute raises error when there are insufficient matches."""
        image1, image2 = simple_images

        task = RegistrationTask(RegistrationMethod.ORB)

        # This may or may not raise an error depending on whether enough features are found
        try:
            result = task.execute(image1, image2)
            assert isinstance(result, RegistrationResult)
        except ValueError as e:
            assert any(msg in str(e) for msg in (
                "Insufficient matches", "No descriptors", "No keypoints", "Failed to compute homography",
            ))

    def test_execute_no_keypoints(self):
        """Test that execute raises error when no keypoints are found."""
        # Create completely blank images
        image1 = np.zeros((100, 100), dtype=np.uint8)
        image2 = np.zeros((100, 100), dtype=np.uint8)

        task = RegistrationTask(RegistrationMethod.ORB)

        with pytest.raises(ValueError, match="No keypoints|No descriptors"):
            task.execute(image1, image2)

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises error in execute."""
        task = RegistrationTask(RegistrationMethod.ORB)
        # Manually set invalid method to test error handling
        task.method = "INVALID"

        image1 = np.zeros((100, 100), dtype=np.uint8)
        image2 = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises(ValueError, match="Unknown method"):
            task.execute(image1, image2)
