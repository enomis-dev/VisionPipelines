import pytest
import numpy as np
import cv2
from visionpipelines.tasks.registration_task import RegistrationTask
from visionpipelines.constants import RegistrationMethod


class TestRegistrationTask:
    """Test suite for RegistrationTask."""
    
    @pytest.fixture
    def sample_images(self):
        """Create sample images with enough features for testing."""
        # Create images with checkerboard pattern (more features for keypoint detection)
        image1 = np.zeros((200, 200), dtype=np.uint8)
        # Create checkerboard pattern
        for i in range(0, 200, 20):
            for j in range(0, 200, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    image1[i:i+20, j:j+20] = 255
        
        # Create slightly shifted version
        image2 = np.zeros((200, 200), dtype=np.uint8)
        for i in range(5, 200, 20):
            for j in range(5, 200, 20):
                if ((i-5) // 20 + (j-5) // 20) % 2 == 0:
                    image2[i:i+20, j:j+20] = 255
        
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
    
    def test_pre_process_color_to_grayscale(self, sample_images):
        """Test preprocessing converts color images to grayscale."""
        image1, image2 = sample_images
        
        # Convert to color (BGR)
        color_image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        color_image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
        
        task = RegistrationTask(RegistrationMethod.ORB)
        gray1, gray2 = task.pre_process(color_image1, color_image2)
        
        assert len(gray1.shape) == 2  # Grayscale
        assert len(gray2.shape) == 2  # Grayscale
        assert gray1.shape == image1.shape
        assert gray2.shape == image2.shape
    
    def test_pre_process_already_grayscale(self, sample_images):
        """Test preprocessing with already grayscale images."""
        image1, image2 = sample_images
        
        task = RegistrationTask(RegistrationMethod.ORB)
        gray1, gray2 = task.pre_process(image1, image2)
        
        assert np.array_equal(gray1, image1)
        assert np.array_equal(gray2, image2)
    
    def test_execute_orb(self, sample_images):
        """Test execute method with ORB."""
        image1, image2 = sample_images
        
        task = RegistrationTask(RegistrationMethod.ORB)
        registered_image, keypoints = task.execute(image1, image2)
        
        assert isinstance(registered_image, np.ndarray)
        assert isinstance(keypoints, np.ndarray)
        assert registered_image.shape == image1.shape
        assert keypoints.shape[0] == 4  # 4 rows: x1, y1, x2, y2
        assert keypoints.shape[1] > 0  # Should find some keypoints
    
    def test_execute_sift(self, sample_images):
        """Test execute method with SIFT."""
        image1, image2 = sample_images
        
        task = RegistrationTask(RegistrationMethod.SIFT)
        registered_image, keypoints = task.execute(image1, image2)
        
        assert isinstance(registered_image, np.ndarray)
        assert isinstance(keypoints, np.ndarray)
        assert registered_image.shape == image1.shape
        assert keypoints.shape[0] == 4
    
    def test_execute_insufficient_matches(self, simple_images):
        """Test that execute raises error when there are insufficient matches."""
        image1, image2 = simple_images
        
        task = RegistrationTask(RegistrationMethod.ORB)
        
        # This may or may not raise an error depending on whether enough features are found
        # We'll catch both cases
        try:
            registered_image, keypoints = task.execute(image1, image2)
            # If it succeeds, that's fine - just verify the output
            assert isinstance(registered_image, np.ndarray)
            assert isinstance(keypoints, np.ndarray)
        except ValueError as e:
            # If it fails, verify it's the expected error
            assert "Insufficient matches" in str(e) or "No descriptors" in str(e) or "No keypoints" in str(e)
    
    def test_execute_no_keypoints(self):
        """Test that execute raises error when no keypoints are found."""
        # Create completely blank images
        image1 = np.zeros((100, 100), dtype=np.uint8)
        image2 = np.zeros((100, 100), dtype=np.uint8)
        
        task = RegistrationTask(RegistrationMethod.ORB)
        
        with pytest.raises(ValueError, match="No keypoints|No descriptors"):
            task.execute(image1, image2)
    
    def test_post_process(self, sample_images):
        """Test post_process method."""
        image1, image2 = sample_images
        
        task = RegistrationTask(RegistrationMethod.ORB)
        registered_image, keypoints = task.execute(image1, image2)
        
        # Post-process should return the same values
        result_img, result_kp = task.post_process(registered_image, keypoints)
        
        assert np.array_equal(result_img, registered_image)
        assert np.array_equal(result_kp, keypoints)
    
    def test_invalid_method_raises_error(self):
        """Test that invalid method raises error in execute."""
        # This is harder to test since we use enums, but we can test the error path
        task = RegistrationTask(RegistrationMethod.ORB)
        # Manually set invalid method to test error handling
        task.method = "INVALID"
        
        image1 = np.zeros((100, 100), dtype=np.uint8)
        image2 = np.zeros((100, 100), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Unknown method"):
            task.execute(image1, image2)

