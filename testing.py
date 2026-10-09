import os
import cv2
import torch
from ultralytics import YOLO

def run_video_inference(video_path, weights_path, output_path=None, conf_threshold=0.4):
    # 1. Check GPU availability
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[+] Running inference on device: {device} ({'GPU' if device == 0 else 'CPU'})")

    # 2. Load fine-tuned weights
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ Weights file not found at: {weights_path}")
    
    model = YOLO(weights_path)
    print(f"[+] Loaded model weights successfully from: {weights_path}")

    # 3. Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Could not open video source: {video_path}")

    # 4. Set up VideoWriter if output saving is requested
    writer = None
    if output_path:
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"[+] Saving annotated output to: {output_path}")

    print("[+] Starting stream... Press 'q' to quit early.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run model inference on current frame
        results = model.predict(source=frame, conf=conf_threshold, device=device, verbose=False)

        # Plot bounding boxes, class names, and confidence scores directly onto frame
        annotated_frame = results[0].plot()

        # Write frame to file if output path set
        if writer:
            writer.write(annotated_frame)

        # Display output in a window
        cv2.imshow("YOLO11 Activity Detection Stream", annotated_frame)

        # Press 'q' to break stream early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[+] Video processing complete.")

if __name__ == "__main__":
    # Adjust paths according to your local folder structure
    INPUT_VIDEO = r"C:\Users\gohar.ali\Desktop\TestCV\Prod2.mp4"
    WEIGHTS_PATH = r"C:\Users\gohar.ali\Desktop\TestCV\best.pt"
    OUTPUT_VIDEO = r"C:\Users\gohar.ali\Desktop\TestCV\annotated_output.mp4"

    run_video_inference(
        video_path=INPUT_VIDEO,
        weights_path=WEIGHTS_PATH,
        output_path=OUTPUT_VIDEO,
        conf_threshold=0.45  # Ignore detections below 45% confidence
    )