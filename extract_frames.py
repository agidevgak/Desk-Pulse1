import os
import cv2
from tqdm import tqdm
from utils import ensure_dir

# Use relative path or forward slashes / raw string
VIDEO_PATH = "C:/Users/gohar.ali/Desktop/Task/Malir-9_ch15_20260918100042_20260918101457.mp4"
OUTPUT_FOLDER = "output_frames/fixed_1000_frames"
TARGET_FRAME_COUNT = 1000

def extract_exact_count():
    ensure_dir(OUTPUT_FOLDER)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Unable to open video file '{VIDEO_PATH}'.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < TARGET_FRAME_COUNT:
        print(f"Warning: Video only has {total_frames} total frames. Extracting all available.")
        frame_interval = 1
    else:
        frame_interval = max(1, total_frames // TARGET_FRAME_COUNT)

    print(f"Total Video Frames: {total_frames}")
    print(f"Extracting 1 frame every {frame_interval} frames to hit ~{TARGET_FRAME_COUNT} target frames...")

    saved_count, frame_count = 0, 0
    pbar = tqdm(total=total_frames, desc="Extracting Frames")

    while cap.isOpened() and saved_count < TARGET_FRAME_COUNT:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            save_path = os.path.join(OUTPUT_FOLDER, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(save_path, frame)
            saved_count += 1

        frame_count += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    print(f"\nDone! Successfully saved {saved_count} frames to '{OUTPUT_FOLDER}/'")

if __name__ == "__main__":
    extract_exact_count()