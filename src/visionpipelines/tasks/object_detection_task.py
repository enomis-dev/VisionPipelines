import cv2
import torch
import numpy as np
from typing import Union, Tuple, List
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from visionpipelines.tasks.task import Task
from visionpipelines.constants import DetectionMethod, Labels

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

    def execute(self, image: torch.Tensor) -> List[dict]:
        """
        Execute object detection on the preprocessed image.
        
        Args:
            image: Preprocessed image tensor.
            
        Returns:
            List of detection dictionaries from the model.
        """
        if self.method == DetectionMethod.FASTER_RCNN:
            return self._detect_faster_rcnn(image)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _detect_faster_rcnn(self, image: torch.Tensor) -> List[dict]:
        """Detect objects using the Faster R-CNN method."""
        # Inference
        with torch.no_grad():
            outputs = self.model(image)
        return outputs

    def pre_process(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess the image for the model input.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV).
            
        Returns:
            Preprocessed image as torch tensor.
        """
        # Convert image to RGB if it's in BGR (as OpenCV loads images in BGR format)
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(self.device)
        return image_tensor

    def post_process(self, outputs: List[dict], threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Post-process the output of the object detection task.
        
        Args:
            outputs: List of detection dictionaries from the model.
            threshold: Confidence threshold for filtering detections.
            
        Returns:
            Tuple of (filtered_boxes, filtered_labels, filtered_scores) as numpy arrays.
        """
        # Outputs is a list of dictionaries, each containing the detections for one image
        output = outputs[0]
        scores = output['scores'].cpu().numpy()
        boxes = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy()

        # Filter by threshold
        mask = scores >= threshold
        filtered_boxes = boxes[mask]
        filtered_labels = labels[mask]
        filtered_scores = scores[mask]

        return filtered_boxes, filtered_labels, filtered_scores

    def draw_boxes(self, image, boxes, labels, scores):
        """ Visualize results with bounding boxes and labels"""
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            label = Labels.COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
            score = scores[i]

            # Draw the bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Add label and score text
            text = f"{label}: {score:.2f}"
            cv2.putText(image, text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 2)

        return image
