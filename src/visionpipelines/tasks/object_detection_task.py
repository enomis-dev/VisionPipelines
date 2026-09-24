import cv2
import torch
import numpy as np
from typing import Optional, Tuple, List
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    SSD300_VGG16_Weights,
    fasterrcnn_resnet50_fpn,
    ssd300_vgg16,
)
from torchvision.transforms import functional as F
from visionpipelines.tasks.task import Task
from visionpipelines.constants import DetectionMethod

DEFAULT_YOLO_WEIGHTS = "yolo11n.pt"


class ObjectDetectionTask(Task):
    def __init__(
        self,
        method: DetectionMethod = DetectionMethod.FASTER_RCNN,
        model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the object detection task with the detection method and optionally the model to use.

        :param method: The method used for object detection (DetectionMethod).
        :param model: Optional, a pre-trained object detection model. If not provided, a default model is loaded
                      based on the selected method. For YOLO this must be an ``ultralytics.YOLO`` instance.
        :param device: Device to run inference on. Defaults to CUDA if available, else CPU.
        """
        self.method = method
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model if model is not None else self.load_default_model(method)

        if method == DetectionMethod.YOLO:
            # Ultralytics models override nn.Module.train(), so calling .eval() on them starts training.
            # The device is passed at predict time instead.
            self.categories = [self.model.names[i] for i in range(len(self.model.names))]
        else:
            self.categories = self._default_weights(method).meta["categories"]
            self.model.to(self.device)
            self.model.eval()

    def _default_weights(self, method: DetectionMethod):
        """Return the pretrained torchvision weights enum used for the given method."""
        if method == DetectionMethod.FASTER_RCNN:
            return FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        elif method == DetectionMethod.SSD:
            return SSD300_VGG16_Weights.DEFAULT
        else:
            raise ValueError(f"Unknown method: {method}")

    def load_default_model(self, method: DetectionMethod) -> torch.nn.Module:
        """
        Load the default model based on the selected detection method.

        :param method: The method used for object detection (DetectionMethod).
        :return: The loaded pre-trained model.
        """
        if method == DetectionMethod.FASTER_RCNN:
            return fasterrcnn_resnet50_fpn(weights=self._default_weights(method))
        elif method == DetectionMethod.SSD:
            return ssd300_vgg16(weights=self._default_weights(method))
        elif method == DetectionMethod.YOLO:
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise ImportError(
                    "YOLO detection requires the 'ultralytics' package. "
                    "Install it with: pip install visionpipelines[yolo]"
                ) from e
            return YOLO(DEFAULT_YOLO_WEIGHTS)
        else:
            raise ValueError(f"Unknown method: {method}")

    def execute(self, image) -> List[dict]:
        """
        Execute object detection on the preprocessed image.

        Args:
            image: Preprocessed image (tensor for torchvision methods, BGR numpy array for YOLO).

        Returns:
            List of detection dictionaries with 'boxes', 'labels' and 'scores' tensors.
        """
        if self.method in (DetectionMethod.FASTER_RCNN, DetectionMethod.SSD):
            return self._detect_torchvision(image)
        elif self.method == DetectionMethod.YOLO:
            return self._detect_yolo(image)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _detect_torchvision(self, image: torch.Tensor) -> List[dict]:
        """Detect objects using a torchvision detection model (Faster R-CNN, SSD)."""
        with torch.no_grad():
            outputs = self.model(image)
        return outputs

    def _detect_yolo(self, image: np.ndarray) -> List[dict]:
        """Detect objects using an Ultralytics YOLO model, converted to the torchvision output format."""
        # conf=0 keeps all candidates so that post_process applies the confidence threshold
        results = self.model.predict(image, device=str(self.device), conf=0.0, verbose=False)
        boxes = results[0].boxes
        return [{
            'boxes': boxes.xyxy,
            'labels': boxes.cls.long(),
            'scores': boxes.conf,
        }]

    def pre_process(self, image: np.ndarray):
        """
        Preprocess the image for the model input.

        Args:
            image: Input image as numpy array (BGR format from OpenCV).

        Returns:
            Preprocessed image as torch tensor, or a 3-channel BGR numpy array for YOLO
            (which handles its own resizing and normalization).
        """
        if self.method == DetectionMethod.YOLO:
            return image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Convert image to RGB if it's in BGR (as OpenCV loads images in BGR format)
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

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
            Labels index into ``self.categories``.
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
            label = self.categories[labels[i]]
            score = scores[i]

            # Draw the bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Add label and score text
            text = f"{label}: {score:.2f}"
            cv2.putText(image, text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 2)

        return image
