import cv2
import torch
import kornia
import numpy as np
from typing import List, Callable, Union, Any, Tuple
from abc import ABC, abstractmethod

class VisionPipeline(ABC):
    """
    Base class for vision processing pipelines.
    
    This class provides a framework for building pipelines that process images
    through a sequence of operations. Subclasses should implement the specific
    pipeline logic for their use case.
    """
    
    def __init__(self):
        """Initialize the pipeline."""
        self._initialized = False
    
    @abstractmethod
    def run_pipeline(self, *args, **kwargs) -> Any:
        """
        Run the complete pipeline on input data.
        
        This method should be implemented by subclasses to define the specific
        pipeline execution logic.
        
        Returns:
            The result of the pipeline execution.
        """
        pass
    
    def _validate_input(self, image: np.ndarray) -> np.ndarray:
        """
        Validate and normalize input image.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            Validated image array.
            
        Raises:
            ValueError: If image is invalid.
        """
        if image is None:
            raise ValueError("Input image cannot be None")
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Input must be numpy array, got {type(image)}")
        if image.size == 0:
            raise ValueError("Input image cannot be empty")
        return image


class TaskBasedPipeline(VisionPipeline):
    """
    Pipeline that processes images through a sequence of Task objects.
    
    This pipeline type uses Task objects that encapsulate preprocessing,
    execution, and postprocessing logic.
    """
    
    def __init__(self, task=None):
        """
        Initialize the task-based pipeline.
        
        Args:
            task: Optional Task object to use in the pipeline.
        """
        super().__init__()
        self.task = task
        self._initialized = True
    
    def run_pipeline(self, *args, **kwargs) -> Any:
        """
        Run the pipeline: preprocess -> execute -> postprocess.
        
        Args:
            *args: Positional arguments to pass to the task.
            **kwargs: Keyword arguments to pass to the task.
            
        Returns:
            The result of the pipeline execution.
        """
        if self.task is None:
            raise ValueError("No task has been set for this pipeline")
        
        # Preprocess
        preprocessed = self.task.pre_process(*args, **kwargs)
        
        # Execute
        if isinstance(preprocessed, tuple):
            result = self.task.execute(*preprocessed, **kwargs)
        else:
            result = self.task.execute(preprocessed, **kwargs)
        
        # Postprocess
        if isinstance(result, tuple):
            return self.task.post_process(*result, **kwargs)
        else:
            return self.task.post_process(result, **kwargs)


class FunctionBasedPipeline(VisionPipeline):
    """
    Pipeline that processes images through a sequence of callable functions.
    
    This pipeline type is useful for simple transformations that don't require
    the full Task abstraction.
    """
    
    def __init__(self):
        """Initialize the function-based pipeline."""
        super().__init__()
        self.tasks: List[Callable] = []
        self._initialized = True
    
    def add_task(self, task: Callable):
        """
        Add a callable function to the pipeline.
        
        Args:
            task: A callable function that processes the input.
        """
        if not callable(task):
            raise TypeError(f"Task must be callable, got {type(task)}")
        self.tasks.append(task)
    
    def run_pipeline(self, image: np.ndarray) -> Any:
        """
        Run all tasks in sequence on the input image.
        
        Args:
            image: Input image as numpy array.
            
        Returns:
            The result after processing through all tasks.
        """
        image = self._validate_input(image)
        
        result = image
        for task in self.tasks:
            result = task(result)
        return result
    
    def clear_pipeline(self):
        """Clear all tasks from the pipeline."""
        self.tasks.clear()
