from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.webcam_pipeline import _aggregate_top_k, run_webcam_pipeline


def test_aggregate_top_k_averages_probs():
    history = deque(maxlen=3)
    history.append([("apple", 0.9), ("banana", 0.1)])
    history.append([("apple", 0.7), ("carrot", 0.3)])
    top = _aggregate_top_k(history, k=3)
    # apple should lead with the averaged probability
    assert top[0][0] == "apple"
    assert pytest.approx(top[0][1], rel=1e-3) == 0.8
    labels = {label for label, _ in top}
    assert {"apple", "banana", "carrot"}.issuperset(labels)


@patch("src.webcam_pipeline.predict")
@patch("src.webcam_pipeline.load_model_and_processor")
@patch("cv2.VideoCapture")
@patch("cv2.imshow")
@patch("cv2.waitKey")
@patch("cv2.setMouseCallback")
@patch("cv2.namedWindow")
def test_run_webcam_pipeline(
    mock_namedWindow,
    mock_setMouseCallback,
    mock_waitKey,
    mock_imshow,
    mock_VideoCapture,
    mock_load_model_and_processor,
    mock_predict,
):
    mock_load_model_and_processor.return_value = ("model", "processor", "cpu")
    mock_predict.return_value = [("apple", 0.95), ("banana", 0.03), ("carrot", 0.02)]

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.read.side_effect = [
        (True, fake_frame),
        (False, None),
    ]
    mock_VideoCapture.return_value = mock_cap

    mock_waitKey.side_effect = [ord("q")]

    try:
        run_webcam_pipeline()
    except Exception as e:
        pytest.fail(f"run_webcam_pipeline raised an exception: {e}")

    mock_VideoCapture.assert_called_once_with(0)
    mock_cap.read.assert_called()
    mock_imshow.assert_called()
    mock_waitKey.assert_called()
    mock_setMouseCallback.assert_called()
    mock_namedWindow.assert_called()
    mock_predict.assert_called()
