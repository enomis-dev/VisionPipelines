from enum import Enum

class TaskType(Enum):
    SEGMENTATION = 1
    DETECTION = 2
    REGISTRATION = 3
    OPTICAL_FLOW = 4

class RegistrationMethod(Enum):
    ORB = "ORB"
    SIFT = "SIFT"


class DetectionMethod(Enum):
    YOLO = "YOLO"
    FASTER_RCNN = "FASTER_RCNN"
    SSD = "SSD"


class SegmentationMethod(Enum):
    DEEPLABV3 = "DEEPLABV3"
    FCN = "FCN"


class OpticalFlowMethod(Enum):
    FARNEBACK = "FARNEBACK"
    RAFT_SMALL = "RAFT_SMALL"
    RAFT_LARGE = "RAFT_LARGE"
