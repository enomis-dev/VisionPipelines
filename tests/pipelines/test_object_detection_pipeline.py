import pytest
import numpy as np
import cv2
from visionpipelines.pipelines.object_detection_pipeline import ObjectDetectionPipeline
from visionpipelines.constants import DetectionMethod


class TestObjectDetectionPipeline:
    """Test suite for ObjectDetectionPipeline."""
    
    @pytest.fixture
    def sample_image(self):
        """Load a sample image for testing."""
        # Try to load the test image if available
        image_path = 'tests/data/people_image.webp'
        image = cv2.imread(image_path)
        
        # If image not found, create a dummy image
        if image is None:
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        return image
    
    def test_pipeline_initialization_faster_rcnn(self):
        """Test pipeline initialization with Faster R-CNN."""
        pipeline = ObjectDetectionPipeline(DetectionMethod.FASTER_RCNN)
        
        assert pipeline.task is not None
        assert pipeline.detector is not None
        assert pipeline.task == pipeline.detector
        assert pipeline.task.method == DetectionMethod.FASTER_RCNN
        assert pipeline.threshold == 0.5  # Default threshold
    
    def test_pipeline_initialization_with_threshold(self):
        """Test pipeline initialization with custom threshold."""
        pipeline = ObjectDetectionPipeline(
            DetectionMethod.FASTER_RCNN,
            threshold=0.7
        )
        
        assert pipeline.threshold == 0.7
    
    def test_pipeline_run_pipeline(self, sample_image):
        """Test running the object detection pipeline."""
        pipeline = ObjectDetectionPipeline(DetectionMethod.FASTER_RCNN)
        
        boxes, labels, scores = pipeline.run_pipeline(sample_image)
        
        assert isinstance(boxes, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert isinstance(scores, np.ndarray)
        
        # All arrays should have the same length
        assert len(boxes) == len(labels) == len(scores)
        
        # If detections found, check shapes
        if len(boxes) > 0:
            assert boxes.shape[1] == 4  # x1, y1, x2, y2
            assert np.all(scores >= 0.5)  # Default threshold
    
    def test_pipeline_run_pipeline_with_custom_threshold(self, sample_image):
        """Test running pipeline with custom threshold."""
        pipeline = ObjectDetectionPipeline(DetectionMethod.FASTER_RCNN)
        
        # Test with high threshold
        boxes1, labels1, scores1 = pipeline.run_pipeline(sample_image, threshold=0.9)
        
        # Test with low threshold
        boxes2, labels2, scores2 = pipeline.run_pipeline(sample_image, threshold=0.1)
        
        # Lower threshold should have same or more detections
        assert len(boxes2) >= len(boxes1)
        
        # Check that scores meet threshold
        if len(scores1) > 0:
            assert np.all(scores1 >= 0.9)
        if len(scores2) > 0:
            assert np.all(scores2 >= 0.1)
    
    def test_pipeline_draw_boxes(self, sample_image):
        """Test drawing boxes on image."""
        pipeline = ObjectDetectionPipeline(DetectionMethod.FASTER_RCNN)
        
        # Run detection
        boxes, labels, scores = pipeline.run_pipeline(sample_image)
        
        # Draw boxes
        if len(boxes) > 0:
            # Only test if we have detections
            result_image = pipeline.draw_boxes(sample_image, boxes, labels, scores)
            
            assert isinstance(result_image, np.ndarray)
            assert result_image.shape == sample_image.shape
            assert result_image.dtype == sample_image.dtype
    
    def test_pipeline_draw_boxes_empty_detections(self, sample_image):
        """Test drawing boxes with no detections."""
        pipeline = ObjectDetectionPipeline(DetectionMethod.FASTER_RCNN)
        
        # Create empty detection arrays
        empty_boxes = np.array([]).reshape(0, 4)
        empty_labels = np.array([], dtype=np.int64)
        empty_scores = np.array([], dtype=np.float32)
        
        # Should not raise error
        result_image = pipeline.draw_boxes(sample_image, empty_boxes, empty_labels, empty_scores)
        
        assert isinstance(result_image, np.ndarray)
        assert result_image.shape == sample_image.shape
