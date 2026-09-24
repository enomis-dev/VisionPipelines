import pytest
import numpy as np
import torch
from visionpipelines.tasks.object_detection_task import ObjectDetectionTask
from visionpipelines.constants import DetectionMethod


class TestObjectDetectionTask:
    """Test suite for ObjectDetectionTask."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        # Create a simple RGB image
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return image
    
    def test_task_initialization_faster_rcnn(self):
        """Test task initialization with Faster R-CNN method."""
        task = ObjectDetectionTask(DetectionMethod.FASTER_RCNN)
        assert task.method == DetectionMethod.FASTER_RCNN
        assert task.model is not None
        assert task.device is not None
    
    def test_task_initialization_with_custom_model(self, sample_image):
        """Test task initialization with custom model."""
        # This would require creating a mock model, but we can test the structure
        task = ObjectDetectionTask(DetectionMethod.FASTER_RCNN, model=None)
        # Should load default model
        assert task.model is not None
    
    def test_pre_process_bgr_to_rgb(self, sample_image):
        """Test preprocessing converts BGR to RGB and creates tensor."""
        task = ObjectDetectionTask(DetectionMethod.FASTER_RCNN)
        
        # Simulate BGR image (OpenCV format)
        bgr_image = sample_image.copy()
        
        tensor = task.pre_process(bgr_image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dim() == 4  # Batch dimension added
        assert tensor.shape[0] == 1  # Batch size of 1
        assert tensor.shape[1] == 3  # RGB channels
    
    def test_pre_process_grayscale(self):
        """Test preprocessing with grayscale image."""
        task = ObjectDetectionTask(DetectionMethod.FASTER_RCNN)
        
        # Create grayscale image
        gray_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        
        tensor = task.pre_process(gray_image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dim() == 4
    
    def test_execute_faster_rcnn(self, sample_image):
        """Test execute method with Faster R-CNN."""
        task = ObjectDetectionTask(DetectionMethod.FASTER_RCNN)
        
        # Preprocess first
        tensor = task.pre_process(sample_image)
        
        # Execute
        outputs = task.execute(tensor)
        
        assert isinstance(outputs, list)
        assert len(outputs) == 1  # One image in batch
        assert isinstance(outputs[0], dict)
        assert 'boxes' in outputs[0]
        assert 'labels' in outputs[0]
        assert 'scores' in outputs[0]
    
    def test_post_process_with_threshold(self, sample_image):
        """Test post_process method with threshold filtering."""
        task = ObjectDetectionTask(DetectionMethod.FASTER_RCNN)
        
        # Preprocess and execute
        tensor = task.pre_process(sample_image)
        outputs = task.execute(tensor)
        
        # Post-process with threshold
        boxes, labels, scores = task.post_process(outputs, threshold=0.5)
        
        assert isinstance(boxes, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert isinstance(scores, np.ndarray)
        
        # All scores should be >= threshold
        if len(scores) > 0:
            assert np.all(scores >= 0.5)
    
    def test_post_process_with_different_threshold(self, sample_image):
        """Test post_process with different threshold values."""
        task = ObjectDetectionTask(DetectionMethod.FASTER_RCNN)
        
        tensor = task.pre_process(sample_image)
        outputs = task.execute(tensor)
        
        # Test with high threshold (should filter more)
        boxes1, labels1, scores1 = task.post_process(outputs, threshold=0.9)
        
        # Test with low threshold (should filter less)
        boxes2, labels2, scores2 = task.post_process(outputs, threshold=0.1)
        
        # Lower threshold should have same or more detections
        assert len(boxes2) >= len(boxes1)
    
    def test_invalid_method_raises_error(self):
        """Test that invalid method raises error."""
        # This is harder to test with enums, but we can test error handling
        task = ObjectDetectionTask(DetectionMethod.FASTER_RCNN)
        task.method = "INVALID"
        
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        tensor = task.pre_process(image)
        
        with pytest.raises(ValueError, match="Unknown method"):
            task.execute(tensor)


class _FakeYoloBoxes:
    def __init__(self):
        self.xyxy = torch.tensor([[10.0, 20.0, 50.0, 80.0], [0.0, 0.0, 5.0, 5.0]])
        self.cls = torch.tensor([0.0, 2.0])
        self.conf = torch.tensor([0.9, 0.3])


class _FakeYoloResult:
    boxes = _FakeYoloBoxes()


class _FakeYolo:
    """Stand-in for ultralytics.YOLO so the YOLO code path can be tested without the package."""
    names = {0: 'person', 1: 'bicycle', 2: 'car'}

    def __init__(self):
        self.predict_kwargs = None

    def predict(self, image, **kwargs):
        self.predict_kwargs = kwargs
        return [_FakeYoloResult()]

    def eval(self):
        raise AssertionError("eval() must not be called on ultralytics models")


class TestObjectDetectionTaskYolo:
    """Test suite for the YOLO code path of ObjectDetectionTask."""

    def test_categories_from_model_names(self):
        task = ObjectDetectionTask(DetectionMethod.YOLO, model=_FakeYolo())
        assert task.categories == ['person', 'bicycle', 'car']

    def test_pre_process_keeps_bgr_numpy(self):
        task = ObjectDetectionTask(DetectionMethod.YOLO, model=_FakeYolo())
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        assert task.pre_process(image) is image

    def test_pre_process_grayscale_to_three_channels(self):
        task = ObjectDetectionTask(DetectionMethod.YOLO, model=_FakeYolo())
        gray = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        assert task.pre_process(gray).shape == (64, 64, 3)

    def test_execute_and_post_process(self):
        model = _FakeYolo()
        task = ObjectDetectionTask(DetectionMethod.YOLO, model=model)
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        outputs = task.execute(task.pre_process(image))
        boxes, labels, scores = task.post_process(outputs, threshold=0.5)

        assert model.predict_kwargs['conf'] == 0.0
        assert boxes.shape == (1, 4)
        assert labels.tolist() == [0]
        assert np.allclose(scores, [0.9])

    def test_draw_boxes_uses_yolo_categories(self):
        task = ObjectDetectionTask(DetectionMethod.YOLO, model=_FakeYolo())
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = np.array([[10, 20, 50, 80]], dtype=np.float32)
        result = task.draw_boxes(image, boxes, np.array([2]), np.array([0.9]))
        assert result.shape == image.shape

    def test_real_yolo_model(self):
        pytest.importorskip("ultralytics")
        task = ObjectDetectionTask(DetectionMethod.YOLO)
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        boxes, labels, scores = task.post_process(task.execute(task.pre_process(image)), threshold=0.5)
        assert len(boxes) == len(labels) == len(scores)


class TestObjectDetectionTaskSsd:
    """Test suite for the SSD code path of ObjectDetectionTask."""

    def test_execute_and_post_process(self):
        task = ObjectDetectionTask(DetectionMethod.SSD)
        assert task.categories[1] == 'person'

        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        outputs = task.execute(task.pre_process(image))
        boxes, labels, scores = task.post_process(outputs, threshold=0.5)

        assert len(boxes) == len(labels) == len(scores)
        if len(scores) > 0:
            assert np.all(scores >= 0.5)


def test_faster_rcnn_categories_match_torchvision_indexing():
    """torchvision detectors use 91-index COCO labels with N/A gaps (13 = stop sign)."""
    task = ObjectDetectionTask(DetectionMethod.FASTER_RCNN)
    assert task.categories[13] == 'stop sign'
    assert len(task.categories) == 91
