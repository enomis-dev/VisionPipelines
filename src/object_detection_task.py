import cv2
import torch
import numpy as np
from typing import Union, Tuple, List
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from src.task import Task
from src.constants import DetectionMethod

class ObjectDetectionTask(Task):
    def __init__(self, method: DetectionMethod = DetectionMethod.FASTER_RCNN, model: Union[torch.nn.Module, None] = None):
        """
        Initialize the object detection task with the detection method and optionally the model to use.

        :param method: The method used for object detection (DetectionMethod).
        :param model: Optional, a pre-trained object detection model. If not provided, a default model is loaded
                      based on the selected method.
        """
        self.method = method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model is None:
            self.model = self.load_default_model(method)
        else:
            self.model = model

        self.model.to(self.device)
        self.model.eval()

    def load_default_model(self, method: DetectionMethod) -> torch.nn.Module:
        """
        Load the default model based on the selected detection method.

        :param method: The method used for object detection (DetectionMethod).
        :return: The loaded pre-trained model.
        """
        if method == DetectionMethod.FASTER_RCNN:
            model = fasterrcnn_resnet50_fpn(pretrained=True)
        else:
            raise ValueError(f"Unknown method: {method}")

        return model

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Perform object detection.

        :param image: The input image as a numpy array (RGB).
        :return: A tuple of the image with bounding boxes drawn and the list of bounding boxes.
                 The bounding boxes are returned as a list of tuples (x_min, y_min, x_max, y_max).
        """
        # Perform object detection
        boxes = self.detect_objects(image)

        # Draw bounding boxes on the image
        image_with_boxes = self.draw_boxes(image, boxes)

        return image_with_boxes, boxes

    def detect_objects(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect objects in the image using the specified method."""
        # Convert image to RGB if it's in BGR (as OpenCV loads images in BGR format)
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        if self.method == DetectionMethod.FASTER_RCNN:
            return self.detect_faster_rcnn(image_rgb)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def detect_faster_rcnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect objects using the Faster R-CNN method."""
        # Convert image to Tensor
        image_tensor = self.preprocess_image(image)

        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Process outputs
        boxes = self.postprocess(outputs)

        return boxes

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess the image for the model input."""
        # Convert the image to a PyTorch tensor and normalize it
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
        return image_tensor

    def postprocess(self, outputs: torch.Tensor) -> List[Tuple[int, int, int, int]]:
        """Post-process the Faster R-CNN outputs to extract bounding boxes."""
        boxes = []
        # Outputs is a list of dictionaries, each containing the detections for one image
        output = outputs[0]
        scores = output['scores'].cpu().numpy()
        boxes_tensor = output['boxes'].cpu().numpy()
        for i, score in enumerate(scores):
            if score > 0.5:  # Confidence threshold
                box = boxes_tensor[i]
                x_min, y_min, x_max, y_max = box
                boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))
        return boxes

    def draw_boxes(self, image: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Draw bounding boxes on the image."""
        for (x_min, y_min, x_max, y_max) in boxes:
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        return image
