import torch
import numpy as np
from typing import List, Tuple
from visionpipelines.tasks.object_detection_task import ObjectDetectionTask
from visionpipelines.constants import DetectionMethod
from visionpipelines.pipelines.vision_pipeline import TaskBasedPipeline


class ObjectDetectionPipeline(TaskBasedPipeline):
    """
    Pipeline for object detection in images.
    
    This pipeline detects objects in images using various detection methods
    (e.g., Faster R-CNN) and returns bounding boxes, labels, and scores.
    """
    
    def __init__(
        self, 
        method: DetectionMethod, 
        model: torch.nn.Module = None, 
        device: torch.device = torch.device('cpu'),
        threshold: float = 0.5
    ):
        """
        Initialize the object detection pipeline.
        
        Args:
            method: The detection method to use (DetectionMethod enum).
            model: Optional pre-trained model. If None, a default model is loaded.
            device: Device to run inference on.
            threshold: Confidence threshold for filtering detections.
        """
        task = ObjectDetectionTask(method=method, model=model)
        super().__init__(task=task)
        self.detector = task  # Keep for backward compatibility
        self.threshold = threshold

    def run_pipeline(self, image: np.ndarray, threshold: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the object detection pipeline on an image.
        
        Args:
            image: Input image as numpy array.
            threshold: Optional confidence threshold (overrides default if provided).
            
        Returns:
            Tuple of (boxes, labels, scores) as numpy arrays.
        """
        # Use provided threshold or default
        thresh = threshold if threshold is not None else self.threshold
        
        # Run the base pipeline (preprocess -> execute)
        outputs = self.task.execute(self.task.pre_process(image))
        
        # Post-process with threshold
        boxes, labels, scores = self.task.post_process(outputs, threshold=thresh)
        
        return boxes, labels, scores

    def draw_boxes(
        self, 
        image: np.ndarray, 
        boxes: np.ndarray, 
        labels: np.ndarray, 
        scores: np.ndarray
    ) -> np.ndarray:
        """
        Draw bounding boxes on the image.
        
        Args:
            image: Input image to draw on.
            boxes: Array of bounding boxes.
            labels: Array of class labels.
            scores: Array of confidence scores.
            
        Returns:
            Image with bounding boxes drawn.
        """
        return self.detector.draw_boxes(image, boxes, labels, scores)
