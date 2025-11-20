import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.webcam_pipeline import run_webcam_pipeline

@patch("cv2.VideoCapture")
@patch("cv2.imshow")
@patch("cv2.waitKey")
def test_run_webcam_pipeline(mock_waitKey, mock_imshow, mock_VideoCapture):
    # Create a fake frame (numpy array) for cv2.cvtColor
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)    
    
    # Mock the VideoCapture object
    mock_cap = MagicMock()
    mock_cap.read.side_effect = [
        (True, fake_frame),  # one valid frame
        (False, None)        # then stop
    ]
    mock_VideoCapture.return_value = mock_cap

    # Mock waitKey to simulate pressing 'q'
    mock_waitKey.side_effect = [ord('q')]

    # Run the pipeline
    try:
        run_webcam_pipeline()
    except Exception as e:
        pytest.fail(f"run_webcam_pipeline raised an exception: {e}")

    # Assertions
    mock_VideoCapture.assert_called_once_with(0)  # Ensure the webcam was opened
    mock_cap.read.assert_called()  # Ensure frames were read
    mock_imshow.assert_called()  # Ensure frames were displayed
    mock_waitKey.assert_called()  # Ensure key press was checked