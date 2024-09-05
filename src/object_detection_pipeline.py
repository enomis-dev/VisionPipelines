import torch
from src.object_detection_task import ObjectDetectionTask
from src.constants import DetectionMethod


class ObjectDetectionPipeline(VisionPipeline):
    def __init__(self, method, model: torch.nn.Module = None, device: torch.device = torch.device('cpu')):
        """Initialize the object detection pipeline with a detection model."""
        super().__init__()
        self.detector = ObjectDetectionTask(method=method, model=model)

    def add_detection_task(self, image = np.ndarray):
        """Add the object detection task to the pipeline."""
        self.tasks.append(self.pre_process(image))
        self.tasks.append(self.detect_objects(image))
        self.tasks.append(self.post_process(image))

    def detect_objects(self, image: torch.Tensor) -> List[Tuple[int, int, int, int]]:
        """Detect objects in the image using the specified method."""
        detector = self.detector.preprocess_image(image)

    def pre_process(self, image: np.ndarray) -> torch.Tensor:
        image_tensor = self.detector.preprocess_image(image)
        return image_tensor

    def post_process(self, outputs: torch.Tensor) -> List[Tuple[int, int, int, int]]:
        """Post-process the detection outputs to extract bounding boxes."""
        boxes = self.detector.post_process(outputs)
        return boxes
