import time
from collections import deque
from typing import List, Tuple

import cv2
from PIL import Image

from src.inference import load_model_and_processor, predict


def _aggregate_top_k(history: deque, k: int = 3) -> List[Tuple[str, float]]:
    """Return top-k labels averaged across the prediction history."""
    if not history:
        return []

    totals = {}
    counts = {}
    for preds in history:
        for label, prob in preds:
            totals[label] = totals.get(label, 0.0) + prob
            counts[label] = counts.get(label, 0) + 1

    averaged = [(label, totals[label] / counts[label]) for label in totals]
    averaged.sort(key=lambda x: x[1], reverse=True)
    return averaged[:k]


def run_webcam_pipeline(
    decision_seconds: float = 7.0,
    confidence_threshold: float = 0.95,
    history_size: int = 30,
):
    """
    Runs the webcam pipeline with a simple UI:
    - Continuously predicts top-3 items from the camera feed.
    - After either a time window or confidence threshold, shows buttons below the video
      for the user to click the correct prediction or "None".
    """

    model, processor, device = load_model_and_processor()

    cap = cv2.VideoCapture(0)
    window_name = "Fruit & Vegetable Classification"
    cv2.namedWindow(window_name)

    prediction_history = deque(maxlen=history_size)
    selection_active = False
    frozen_options: List[Tuple[str, float]] = []
    last_decision_time = time.time()
    button_regions = []

    def on_mouse(event, x, y, flags, param):
        nonlocal selection_active, frozen_options, last_decision_time, prediction_history
        if event != cv2.EVENT_LBUTTONDOWN or not selection_active:
            return
        for label, (x0, y0, x1, y1) in button_regions:
            if x0 <= x <= x1 and y0 <= y <= y1:
                print(f"User selected: {label}")
                prediction_history.clear()
                selection_active = False
                frozen_options = []
                last_decision_time = time.time()
                break

    cv2.setMouseCallback(window_name, on_mouse)
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        frame_height, frame_width = frame.shape[:2]

        if not selection_active:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            predictions = predict(image, model, processor, device, top_k=3)
            prediction_history.append(predictions)

            averaged = _aggregate_top_k(prediction_history, k=3)
            if averaged:
                top_label, top_prob = averaged[0]
                now = time.time()
                if top_prob >= confidence_threshold or (now - last_decision_time) >= decision_seconds:
                    selection_active = True
                    frozen_options = averaged
        else:
            averaged = frozen_options

        # Draw prediction text
        text = ", ".join([f"{label}: {prob:.2%}" for label, prob in averaged]) if averaged else "Collecting predictions..."
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Prepare buttons (top-3 + None)
        labels = [label for label, _ in (frozen_options or averaged)] if averaged else []
        labels = labels[:3]  # ensure at most three
        labels.append("None")

        button_height = 60
        y0 = max(frame_height - button_height - 10, 0)
        spacing = 10
        button_width = int((frame_width - spacing * (len(labels) + 1)) / max(len(labels), 1))
        button_regions = []

        for idx, label in enumerate(labels):
            x0 = spacing + idx * (button_width + spacing)
            y1 = frame_height - 10
            x1 = x0 + button_width
            color = (225, 225, 225) if selection_active else (100, 100, 100)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x0 + 10, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )
            button_regions.append((label, (x0, y0, x1, y1)))

        prompt = "Click the correct item or 'None'" if selection_active else "Gathering confidence..."
        cv2.putText(frame, prompt, (10, frame_height - button_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
