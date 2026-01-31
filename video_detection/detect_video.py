# NOTE:
# This script is optional and not used in the web-based implementation.
# Video detection is handled via frame extraction in backend/predict.py

import cv2
from backend.predict import predict_image

def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    results = {
        "REAL IMAGE": 0,
        "DEEPFAKE / MANIPULATED": 0,
        "AI-GENERATED (SYNTHETIC)": 0,
        "UNCERTAIN (AI-LIKE)": 0
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process 1 frame per second
        if frame_count % 30 == 0:
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)

            label, _ = predict_image(temp_path)
            if label in results:
                results[label] += 1

        frame_count += 1

    cap.release()

    final_label = max(results, key=results.get)

    print("Frame-wise results:", results)
    print("FINAL VIDEO RESULT:", final_label)

# Example usage
detect_video("sample_video.mp4")
