from abc import ABC, abstractmethod
import numpy as np

class Task(ABC):
    """
    An abstract base class for tasks involving image processing.
    """

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Perform the task on the given image.

        :param image: The image to process, as a numpy array.
        :return: The processed image, as a numpy array.
        """
        pass
