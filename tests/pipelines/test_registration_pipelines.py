import pytest
import cv2
import matplotlib
from visionpipelines import RegistrationPipeline, RegistrationResult
from visionpipelines.constants import RegistrationMethod

matplotlib.use("Agg")


@pytest.fixture
def setup_images():
    """Fixture to load test images."""
    image1 = cv2.imread('tests/data/IMG1_low_res.jpg')
    image2 = cv2.imread('tests/data/IMG2_low_res.jpg')

    # Ensure images were loaded
    assert image1 is not None, "Failed to load IMG1_low_res.jpg"
    assert image2 is not None, "Failed to load IMG2_low_res.jpg"

    return image1, image2


@pytest.mark.parametrize("method", [RegistrationMethod.ORB, RegistrationMethod.SIFT])
def test_registration_pipeline(setup_images, method):
    """Test registration pipeline end to end on real images."""
    image1, image2 = setup_images

    pipeline = RegistrationPipeline(method)
    result = pipeline.run_pipeline(image1, image2)

    assert isinstance(result, RegistrationResult)
    # Registered image is in image1's frame and keeps its color channels
    assert result.registered_image.shape == image1.shape
    assert result.transform.shape == (3, 3)
    assert result.matches.shape[1] == 4
    assert result.matches.shape[0] > 0
    assert 0.0 < result.inlier_ratio <= 1.0


def test_registration_pipeline_task_access():
    """Test that the task can be accessed through the pipeline."""
    pipeline = RegistrationPipeline(RegistrationMethod.ORB)

    assert pipeline.task is not None
    assert pipeline.registrator is not None
    assert pipeline.task == pipeline.registrator
    assert pipeline.task.method == RegistrationMethod.ORB


def test_plot_matches(setup_images, monkeypatch):
    """plot_matches accepts the (N, 4) matches from the result."""
    image1, image2 = setup_images
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    pipeline = RegistrationPipeline(RegistrationMethod.ORB)
    result = pipeline.run_pipeline(image1, image2)

    pipeline.plot_matches(image1, image2, result.matches)
