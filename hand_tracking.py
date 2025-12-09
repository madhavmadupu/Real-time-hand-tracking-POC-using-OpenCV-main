"""
Real-time hand tracking and danger detection using classical OpenCV.

Key features:
- Skin detection in YCrCb color space
- Motion-based filtering to reduce false positives
- Distance-based SAFE/WARNING/DANGER state machine
- Keyboard controls: Q/ESC quit, R reset background, S screenshot
"""

import os
from datetime import datetime

import cv2
import numpy as np

# Configuration -------------------------------------------------------------
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

BOUNDARY = {
    "x": 220,
    "y": 60,
    "width": 200,
    "height": 300,
}

DANGER_THRESHOLD = 80
WARNING_THRESHOLD = 150
MIN_CONTOUR_AREA = 2000

LOWER_SKIN_YCRCB = np.array([0, 133, 77], dtype=np.uint8)
UPPER_SKIN_YCRCB = np.array([255, 173, 127], dtype=np.uint8)

KERNEL_ERODE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
KERNEL_DILATE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


# Utility helpers -----------------------------------------------------------
def ensure_output_dirs() -> None:
    os.makedirs("screenshots", exist_ok=True)


def skin_mask(frame: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, LOWER_SKIN_YCRCB, UPPER_SKIN_YCRCB)
    mask = cv2.erode(mask, KERNEL_ERODE, iterations=2)
    mask = cv2.dilate(mask, KERNEL_DILATE, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


def combine_with_motion(skin: np.ndarray, gray: np.ndarray, prev_gray: np.ndarray | None):
    if prev_gray is None:
        return skin, None

    frame_diff = cv2.absdiff(prev_gray, gray)
    _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.erode(motion_mask, KERNEL_ERODE, iterations=1)
    motion_mask = cv2.dilate(motion_mask, KERNEL_DILATE, iterations=1)
    combined = cv2.bitwise_and(skin, motion_mask)
    return combined, motion_mask


def find_hand_contour(mask: np.ndarray, frame_height: int):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Favor lower parts of the frame (hands usually below faces)
        if cy < frame_height * 0.5:
            continue

        score = area * (1 + (cy / frame_height))
        if score > best_score:
            best_score = score
            best = contour

    return best


def distance_to_boundary(x: int, y: int) -> int:
    rect_x, rect_y = BOUNDARY["x"], BOUNDARY["y"]
    rect_w, rect_h = BOUNDARY["width"], BOUNDARY["height"]

    clamped_x = min(max(x, rect_x), rect_x + rect_w)
    clamped_y = min(max(y, rect_y), rect_y + rect_h)
    return int(np.hypot(x - clamped_x, y - clamped_y))


def classify_state(distance: int | None):
    if distance is None:
        return "NO_HAND", (180, 180, 180)
    if distance <= DANGER_THRESHOLD:
        return "DANGER", (0, 0, 255)
    if distance <= WARNING_THRESHOLD:
        return "WARNING", (0, 165, 255)
    return "SAFE", (0, 255, 0)


def draw_boundary(frame: np.ndarray, color: tuple[int, int, int]) -> None:
    start = (BOUNDARY["x"], BOUNDARY["y"])
    end = (BOUNDARY["x"] + BOUNDARY["width"], BOUNDARY["y"] + BOUNDARY["height"])
    cv2.rectangle(frame, start, end, color, 2)


def overlay_state(frame: np.ndarray, state: str, distance: int | None, color: tuple[int, int, int]) -> None:
    text = f"{state}"
    if distance is not None:
        text += f" | dist: {distance}px"
    cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def save_screenshot(frame: np.ndarray) -> str:
    ensure_output_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("screenshots", f"screenshot_{timestamp}.jpg")
    cv2.imwrite(path, frame)
    return path


# Main loop -----------------------------------------------------------------
def main():
    ensure_output_dirs()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to read from camera. Is it connected?")
            break

        frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask_skin = skin_mask(frame)
        mask_combined, motion_mask = combine_with_motion(mask_skin, gray, prev_gray)
        hand_mask = mask_combined if mask_combined is not None else mask_skin

        contour = find_hand_contour(hand_mask, frame.shape[0])

        distance = None
        state_color = (180, 180, 180)
        state = "NO_HAND"

        if contour is not None:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                distance = distance_to_boundary(cx, cy)
                state, state_color = classify_state(distance)

                cv2.drawContours(frame, [contour], -1, state_color, 2)
                cv2.circle(frame, (cx, cy), 6, state_color, -1)
                cv2.putText(frame, "HAND", (cx - 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        else:
            state, state_color = classify_state(None)

        draw_boundary(frame, state_color)
        overlay_state(frame, state, distance, state_color)

        cv2.imshow("Hand Tracking", frame)
        cv2.imshow("Mask", hand_mask)
        if motion_mask is not None:
            cv2.imshow("Motion", motion_mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
        if key == ord("r"):
            prev_gray = None
            print("Reset background reference.")
        if key == ord("s"):
            path = save_screenshot(frame)
            print(f"Saved screenshot to {path}")

        prev_gray = gray

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

