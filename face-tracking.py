from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp


MODEL_PATH = Path(__file__).with_name("face_landmarker.task")
CAMERA_INDEX = 0
MAX_FACES = 1


def draw_landmarks(
    frame,
    face_landmarks,
    radius: int = 1,
) -> None:
    """Draw face landmarks on the frame."""

    height, width = frame.shape[:2]

    for landmark in face_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(
                frame,
                (x, y),
                radius,
                (0, 255, 0),
                -1,
                lineType=cv2.LINE_AA,
            )


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Place face_landmarker.task in the same directory."
        )

    capture = cv2.VideoCapture(CAMERA_INDEX)

    if not capture.isOpened():
        raise RuntimeError(
            "Failed to open the webcam. "
            "Try CAMERA_INDEX = 0, 1, or 2."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=MAX_FACES,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )

    start_time = time.monotonic()

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                success, frame = capture.read()

                if not success:
                    print("Failed to read frame from webcam.")
                    break

                # Mirror the image.
                frame = cv2.flip(frame, 1)

                # Convert BGR to RGB.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int(
                    (time.monotonic() - start_time) * 1000
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                face_count = len(result.face_landmarks)

                for face_landmarks in result.face_landmarks:
                    draw_landmarks(frame, face_landmarks)

                cv2.putText(
                    frame,
                    f"Faces: {face_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    "Press Q to quit",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("MediaPipe Face Tracker", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
