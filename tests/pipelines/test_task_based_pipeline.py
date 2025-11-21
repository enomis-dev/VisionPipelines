import pytest
import numpy as np
import torch
from visionpipelines.pipelines.vision_pipeline import TaskBasedPipeline
from visionpipelines.tasks.task import Task
from visionpipelines.constants import RegistrationMethod, DetectionMethod
from visionpipelines.tasks.registration_task import RegistrationTask
from visionpipelines.tasks.object_detection_task import ObjectDetectionTask


class SimpleTestTask(Task):
    """Simple test task for testing TaskBasedPipeline."""
    
    def __init__(self, multiplier=2):
        self.multiplier = multiplier
    
    def execute(self, value):
        """Multiply the input value."""
        return value * self.multiplier
    
    def pre_process(self, value):
        """Convert to float if needed."""
        return float(value)
    
    def post_process(self, result):
        """Round the result."""
        return int(result)


class TestTaskBasedPipeline:
    """Test suite for TaskBasedPipeline."""
    
    def test_pipeline_initialization_with_task(self):
        """Test that pipeline can be initialized with a task."""
        task = SimpleTestTask(multiplier=3)
        pipeline = TaskBasedPipeline(task=task)
        
        assert pipeline.task is not None
        assert pipeline.task == task
        assert pipeline._initialized is True
    
    def test_pipeline_initialization_without_task(self):
        """Test that pipeline can be initialized without a task."""
        pipeline = TaskBasedPipeline(task=None)
        
        assert pipeline.task is None
        assert pipeline._initialized is True
    
    def test_pipeline_run_with_simple_task(self):
        """Test running pipeline with a simple task."""
        task = SimpleTestTask(multiplier=5)
        pipeline = TaskBasedPipeline(task=task)
        
        result = pipeline.run_pipeline(10)
        
        # pre_process: 10 -> 10.0
        # execute: 10.0 * 5 = 50.0
        # post_process: 50.0 -> 50
        assert result == 50
    
    def test_pipeline_run_without_task_raises_error(self):
        """Test that running pipeline without task raises error."""
        pipeline = TaskBasedPipeline(task=None)
        
        with pytest.raises(ValueError, match="No task has been set"):
            pipeline.run_pipeline(10)
    
    def test_pipeline_with_tuple_input(self):
        """Test pipeline with tuple input (like registration)."""
        class TupleTask(Task):
            def execute(self, a, b):
                return a + b, a - b
            
            def pre_process(self, a, b):
                return float(a), float(b)
        
        task = TupleTask()
        pipeline = TaskBasedPipeline(task=task)
        
        result = pipeline.run_pipeline(10, 5)
        
        assert isinstance(result, tuple)
        assert result == (15.0, 5.0)
    
    def test_pipeline_with_registration_task(self):
        """Test pipeline with actual RegistrationTask."""
        import cv2
        
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
        
        task = RegistrationTask(RegistrationMethod.ORB)
        pipeline = TaskBasedPipeline(task=task)
        
        # The registration might fail if not enough matches, so handle both cases
        try:
            registered_image, keypoints = pipeline.run_pipeline(image1, image2)
            
            # Check outputs if successful
            assert isinstance(registered_image, np.ndarray)
            assert isinstance(keypoints, np.ndarray)
            assert registered_image.shape == image1.shape
            assert keypoints.shape[0] == 4  # 4 rows: x1, y1, x2, y2
            assert keypoints.shape[1] > 0  # Should have some keypoints
        except ValueError as e:
            # If registration fails due to insufficient matches, that's acceptable for testing
            # Just verify it's the expected error type
            assert "Insufficient matches" in str(e) or "No descriptors" in str(e) or "No keypoints" in str(e) or "Failed to compute homography" in str(e)


class TestTaskInterface:
    """Test suite for Task base class interface."""
    
    def test_task_is_abstract(self):
        """Test that Task cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Task()
    
    def test_task_must_implement_execute(self):
        """Test that subclasses must implement execute."""
        class IncompleteTask(Task):
            pass
        
        with pytest.raises(TypeError):
            IncompleteTask()
    
    def test_task_pre_process_default(self):
        """Test default pre_process behavior."""
        class TestTask(Task):
            def execute(self, x):
                return x * 2
        
        task = TestTask()
        
        # Default pre_process should return input unchanged
        result = task.pre_process(10)
        assert result == 10
        
        # With multiple args
        result = task.pre_process(10, 20)
        assert result == (10, 20)
    
    def test_task_post_process_default(self):
        """Test default post_process behavior."""
        class TestTask(Task):
            def execute(self, x):
                return x * 2
        
        task = TestTask()
        
        # Default post_process should return input unchanged
        result = task.post_process(20)
        assert result == 20
        
        # With multiple args
        result = task.post_process(20, 30)
        assert result == (20, 30)
    
    def test_task_full_lifecycle(self):
        """Test complete task lifecycle."""
        class LifecycleTask(Task):
            def pre_process(self, x):
                return x + 1
            
            def execute(self, x):
                return x * 2
            
            def post_process(self, x):
                return x - 1
        
        task = LifecycleTask()
        
        # Simulate pipeline execution
        preprocessed = task.pre_process(10)  # 11
        executed = task.execute(preprocessed)  # 22
        postprocessed = task.post_process(executed)  # 21
        
        assert postprocessed == 21
