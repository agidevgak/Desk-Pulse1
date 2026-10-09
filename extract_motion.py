import os
import cv2
import numpy as np
from tqdm.auto import tqdm
from utils import ensure_dir

VIDEO_PATH = "Malir-9_ch15_20260918100042_20260918101457.mp4" 
OUTPUT_FOLDER = "output_frames/motion_1000_frames"
TARGET_FRAME_COUNT = 1000
MOTION_THRESHOLD = 12.0

def extract_motion_1000():
    ensure_dir(OUTPUT_FOLDER)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Unable to open video file at '{VIDEO_PATH}'. Check file path.")
        return

    raw_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if raw_total > 0:
        total_frames = raw_total
        step_size = max(1, total_frames // (TARGET_FRAME_COUNT * 2))
    else:
        total_frames = None
        step_size = 1

    print(f"Total Video Frames: {total_frames if total_frames else 'Unknown'}")
    print(f"Sampling every {step_size} frames with Motion Filtering...")

    prev_gray = None
    saved_count = 0
    frame_count = 0

    pbar = tqdm(total=total_frames, desc="Extracting Motion Frames")

    try:
        while cap.isOpened() and saved_count < TARGET_FRAME_COUNT:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_count % step_size == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if prev_gray is None:
                    prev_gray = gray
                    save_path = os.path.join(OUTPUT_FOLDER, f"frame_{saved_count:04d}.jpg")
                    cv2.imwrite(save_path, frame)
                    saved_count += 1
                else:
                    frame_diff = cv2.absdiff(prev_gray, gray)
                    
                    # FIX: Use cv2.mean() to prevent uint8 datatype issues
                    diff_score = cv2.mean(frame_diff)[0]

                    if diff_score > MOTION_THRESHOLD:
                        save_path = os.path.join(OUTPUT_FOLDER, f"frame_{saved_count:04d}.jpg")
                        cv2.imwrite(save_path, frame)
                        prev_gray = gray
                        saved_count += 1

            frame_count += 1
            pbar.update(1)
    finally:
        cap.release()
        pbar.close()

    print(f"\nDone! Successfully saved {saved_count} motion-triggered frames to '{OUTPUT_FOLDER}'")

if __name__ == "__main__":
    extract_motion_1000()