import torch
import numpy as np
from typing import List, Tuple
from visionpipelines.tasks.object_detection_task import ObjectDetectionTask
from visionpipelines.constants import DetectionMethod
from visionpipelines.pipelines.vision_pipeline import VisionPipeline


class ObjectDetectionPipeline(VisionPipeline):
    def __init__(self, method, model: torch.nn.Module = None, device: torch.device = torch.device('cpu')):
        """Initialize the object detection pipeline with a detection model."""
        super().__init__()
        self.detector = ObjectDetectionTask(method=method, model=model)

    def add_detection_task(self):
        """Add the object detection task to the pipeline."""
        self.tasks.append(self.pre_process)
        self.tasks.append(self.detect_objects)
        self.tasks.append(self.post_process)

    def detect_objects(self, image: torch.Tensor) -> List[Tuple[int, int, int, int]]:
        """Detect objects in the image using the specified method."""
        objects = self.detector.detect_objects(image)
        return objects

    def pre_process(self, image: np.ndarray) -> torch.Tensor:
        image_tensor = self.detector.pre_process(image)
        return image_tensor

    def post_process(self, outputs: torch.Tensor) -> List[Tuple[int, int, int, int]]:
        """Post-process the detection outputs to extract bounding boxes."""
        boxes = self.detector.post_process(outputs)
        return boxes

    def run_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Perform object detection"""
        self.add_detection_task()
        output = super().run_pipeline(image)
        return output

    def draw_boxes(
        self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray
        ) -> np.ndarray:
        """Draw bounding boxes on the image."""
        return self.detector.draw_boxes(image, boxes, labels, scores)
