from abc import ABC, abstractmethod
from typing import Any


class Task(ABC):
    """
    An abstract base class for tasks involving image processing.
    
    Tasks encapsulate a single image processing operation (e.g., detection, registration).
    They handle preprocessing, the main operation, and postprocessing.
    """
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the main task operation.
        
        This method should be implemented by subclasses to perform the core
        image processing operation.
        
        Returns:
            The result of the task execution (type depends on the specific task).
        """
        pass
    
    def pre_process(self, *args, **kwargs) -> Any:
        """
        Preprocess input data before executing the task.
        
        Default implementation returns inputs unchanged. Override in subclasses
        to add preprocessing logic.
        
        Returns:
            Preprocessed inputs (type depends on the specific task).
        """
        return args if len(args) > 1 else args[0] if args else None
    
    def post_process(self, *args, **kwargs) -> Any:
        """
        Postprocess output data after executing the task.
        
        Default implementation returns outputs unchanged. Override in subclasses
        to add postprocessing logic.
        
        Returns:
            Postprocessed outputs (type depends on the specific task).
        """
        return args if len(args) > 1 else args[0] if args else None

