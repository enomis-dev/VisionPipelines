# VisionPipelines

<div style="text-align: center;">
    <img src="project_image.webp" alt="VisionPipeline Thumbnail" width="400"/>
</div>

VisionPipelines makes it easy to build advanced image-processing pipelines for common computer vision tasks such as image registration and object detection, composed from reusable, testable building blocks.

## Installation

```bash
pip install visionpipelines
```

Or add it to a project managed with [uv](https://docs.astral.sh/uv/):

```bash
uv add visionpipelines
```

## Quickstart

### Object detection

```python
import cv2
from visionpipelines import ObjectDetectionPipeline, DetectionMethod

image = cv2.imread("image.jpg")

pipeline = ObjectDetectionPipeline(DetectionMethod.FASTER_RCNN, threshold=0.7)
boxes, labels, scores = pipeline.run_pipeline(image)

annotated = pipeline.draw_boxes(image, boxes, labels, scores)
```

### Image registration

```python
import cv2
from visionpipelines import RegistrationPipeline, RegistrationMethod

image1 = cv2.imread("reference.jpg")
image2 = cv2.imread("moving.jpg")

pipeline = RegistrationPipeline(RegistrationMethod.ORB)
registered_image, keypoints = pipeline.run_pipeline(image1, image2)

pipeline.plot_matches(image1, image2, keypoints)
```

### Custom function-based pipelines

For simple transformations that don't need the full `Task` abstraction, compose a pipeline out of plain callables:

```python
from visionpipelines import FunctionBasedPipeline
from visionpipelines.utils import resize, to_grayscale

pipeline = FunctionBasedPipeline()
pipeline.add_task(resize((256, 256)))
pipeline.add_task(to_grayscale())

result = pipeline.run_pipeline(image_tensor)
```

## Architecture

- **`Task`** — encapsulates a single operation (`pre_process` → `execute` → `post_process`), e.g. `ObjectDetectionTask`, `RegistrationTask`.
- **`TaskBasedPipeline`** — runs a `Task` through its full lifecycle; used by `ObjectDetectionPipeline` and `RegistrationPipeline`.
- **`FunctionBasedPipeline`** — chains plain callables for lighter-weight transformations.

To add a new capability, implement a `Task` subclass and, if useful, wrap it in a dedicated pipeline.

## Contributing to VisionPipelines

Thank you for your interest in contributing to VisionPipelines! To ensure a smooth development experience and maintain consistency across contributions, please follow the guidelines below.

### Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/visionpipeline.git
   cd visionpipeline
   ```

2. **Set up your development environment**

   VisionPipelines uses [uv](https://docs.astral.sh/uv/) for dependency management and virtual environments.

   - **Install uv** (if you haven't already):

     ```bash
     pip install uv
     ```

   - **Install project dependencies**:

     ```bash
     uv sync
     ```

     This creates a `.venv` and installs the project along with its dev dependencies. Run commands inside it with `uv run <command>`.

### Development and Testing

- **Run tests**: Ensure all tests pass before submitting a contribution.

  ```bash
  uv run pytest
  ```

- **Code style**: Follow [PEP 8](https://pep8.org/) for consistent formatting and readability.

### Making a Contribution

1. **Fork the repository** on GitHub.

2. **Create a feature branch**

   ```bash
   git checkout -b your-feature-branch
   ```

3. **Make your changes**, then commit with a descriptive message:

   ```bash
   git add .
   git commit -m "Describe your changes"
   ```

4. **Push your branch**

   ```bash
   git push origin your-feature-branch
   ```

5. **Submit a pull request** from your fork to the original repository, with a clear description of your changes and why they should be merged.

### Reviewing and Merging

- **Code review**: All pull requests are reviewed by the maintainers, who may request changes.
- **Merge**: Once approved and passing all tests, your pull request is merged into `main`.

### Notebooks

To play with the notebooks, set up your environment as described in "Getting Started", then:

1. **Install Jupyter**

   ```bash
   uv add --dev jupyter ipykernel
   ```

2. **Register the kernel**

   ```bash
   uv run python -m ipykernel install --user --name visionpipelines --display-name "visionpipelines"
   ```

3. **Start Jupyter**

   ```bash
   uv run jupyter notebook
   ```

### Additional Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [PEP 8 Style Guide](https://pep8.org/)

Thank you for contributing to VisionPipelines!
