import pytest
import numpy as np
import cv2
from visionpipelines.tasks.optical_flow_task import OpticalFlowTask
from visionpipelines.constants import OpticalFlowMethod

SHIFT_X, SHIFT_Y = 6, 4
# Ignore the borders, where the shifted image has no content to match
MARGIN = 20


@pytest.fixture(scope="module")
def shifted_images():
    """A real BGR image and a copy translated by a known offset."""
    image1 = cv2.imread('tests/data/IMG1_low_res.jpg')
    assert image1 is not None, "Failed to load IMG1_low_res.jpg"

    height, width = image1.shape[:2]
    shift = np.float32([[1, 0, SHIFT_X], [0, 1, SHIFT_Y]])
    image2 = cv2.warpAffine(image1, shift, (width, height))
    return image1, image2


@pytest.fixture(scope="module", params=list(OpticalFlowMethod), ids=lambda m: m.name)
def task(request):
    """One task per method, shared across tests to avoid reloading RAFT weights."""
    return OpticalFlowTask(request.param)


def _inner(array):
    return array[MARGIN:-MARGIN, MARGIN:-MARGIN]


class TestOpticalFlowTask:
    """Test suite for OpticalFlowTask."""

    def test_flow_shape_and_dtype(self, task, shifted_images):
        image1, image2 = shifted_images

        flow = task.execute(image1, image2)

        assert flow.shape == (*image1.shape[:2], 2)
        assert flow.dtype == np.float32

    def test_recovers_known_shift(self, task, shifted_images):
        """flow[y, x] = (dx, dy) is where the pixel of image1 moved to in image2."""
        image1, image2 = shifted_images

        flow = task.execute(image1, image2)
        median_dx, median_dy = np.median(_inner(flow).reshape(-1, 2), axis=0)

        assert median_dx == pytest.approx(SHIFT_X, abs=0.5)
        assert median_dy == pytest.approx(SHIFT_Y, abs=0.5)

    def test_warp_aligns_image2_with_image1(self, task, shifted_images):
        image1, image2 = shifted_images

        warped = task.warp(image2, task.execute(image1, image2))

        error_before = np.abs(_inner(image2).astype(int) - _inner(image1)).mean()
        error_after = np.abs(_inner(warped).astype(int) - _inner(image1)).mean()
        assert warped.shape == image2.shape
        assert error_after < error_before / 10

    @pytest.mark.parametrize("size", [(101, 77), (64, 64), (203, 137)])
    def test_any_image_size(self, task, shifted_images, size):
        """RAFT needs sizes divisible by 8 and >= 128; inputs are padded and the flow cropped back."""
        height, width = size
        image1, image2 = (image[:height, :width] for image in shifted_images)

        flow = task.execute(image1, image2)

        assert flow.shape == (height, width, 2)

    def test_grayscale_input(self, task, shifted_images):
        image1, image2 = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in shifted_images)

        flow = task.execute(image1, image2)

        assert flow.shape == (*image1.shape, 2)

    def test_mismatched_sizes_raise_error(self, task, shifted_images):
        image1, image2 = shifted_images

        with pytest.raises(ValueError, match="same height and width"):
            task.execute(image1, image2[:-10])

    def test_flow_to_color(self, task, shifted_images):
        image1, image2 = shifted_images

        color = task.flow_to_color(task.execute(image1, image2))

        assert color.shape == image1.shape
        assert color.dtype == np.uint8


def test_farneback_has_no_model():
    task = OpticalFlowTask(OpticalFlowMethod.FARNEBACK)
    assert task.model is None


def test_invalid_method_raises_error():
    with pytest.raises(ValueError, match="Unknown method"):
        OpticalFlowTask("INVALID")
