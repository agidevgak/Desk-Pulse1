import os
import cv2
from tqdm import tqdm
from utils import ensure_dir, select_roi

VIDEO_PATH = "Malir-9_ch15_20260918100042_20260918101457.mp4"
OUTPUT_FOLDER = "output_frames/roi_1000_frames"
TARGET_FRAME_COUNT = 1000

def extract_roi_1000():
    ensure_dir(OUTPUT_FOLDER)

    # 1. Select ROI interactively
    roi = select_roi(VIDEO_PATH)
    if roi is None or roi[2] == 0 or roi[3] == 0:
        print("Invalid ROI selected. Exiting.")
        return

    x, y, w, h = roi
    print(f"Selected ROI Area: X={x}, Y={y}, Width={w}, Height={h}")

    # 2. Extract and crop
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Unable to open video file '{VIDEO_PATH}'.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // TARGET_FRAME_COUNT)

    saved_count, frame_count = 0, 0
    pbar = tqdm(total=total_frames, desc="Extracting Cropped ROI Frames")

    while cap.isOpened() and saved_count < TARGET_FRAME_COUNT:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            cropped_frame = frame[int(y):int(y+h), int(x):int(x+w)]
            save_path = os.path.join(OUTPUT_FOLDER, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(save_path, cropped_frame)
            saved_count += 1

        frame_count += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    print(f"\nDone! Successfully saved {saved_count} cropped frames to '{OUTPUT_FOLDER}/'")

if __name__ == "__main__":
    extract_roi_1000()