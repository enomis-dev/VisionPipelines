from enum import Enum

class TaskType(Enum):
    SEGMENTATION = 1
    DETECTION = 2
    REGISTRATION = 3

class RegistrationMethod(Enum):
    ORB = "ORB"
    SIFT = "SIFT"
