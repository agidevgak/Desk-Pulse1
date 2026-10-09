import os
import cv2
import numpy as np
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
VIDEO_PATHS = [
    "Trim & VF-4 Area K5_ch8_20260918102959_20260918104524.mp4"
]

OUTPUT_FOLDER = "output_frames/dataset_1000_roi"
TARGET_TOTAL_FRAMES = 1200     # Target clean frame count
HASH_DIFF_THRESHOLD = 8         # Adjusted for higher variation capture
FRAME_STEP = 2                  # Evaluate every 2nd frame for maximum sampling density

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def calculate_dhash(image, hash_size=8):
    """Generates a 64-bit difference hash for deduplication."""
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()

def hamming_distance(hash1, hash2):
    """Calculates bit difference between two hashes."""
    return np.count_nonzero(hash1 != hash2)

def select_roi(video_path):
    """Pops up interactive GUI to select the target table/desk ROI."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"[ERROR] Could not read video: {video_path}")
        return None
    
    window_title = f"Select Desk ROI for: {os.path.basename(video_path)} (Press SPACE/ENTER when done)"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(window_title, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_title)
    return roi

# ==============================================================================
# MAIN PROCESSING ENGINE
# ==============================================================================
def process_videos():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    kept_hashes = []
    total_saved = 0

    print("=" * 60)
    print(f" Starting Extraction for: {VIDEO_PATHS[0]}")
    print(f" Target Output: ~{TARGET_TOTAL_FRAMES} Unique Desk ROI Frames")
    print("=" * 60)

    for vid_idx, video_path in enumerate(VIDEO_PATHS):
        if not os.path.exists(video_path):
            print(f"[ERROR] File not found: '{video_path}'. Check if the filename matches exactly.")
            continue

        print(f"\nProcessing Video: {os.path.basename(video_path)}")
        
        # Interactive ROI Selection
        roi = select_roi(video_path)
        if roi is None or roi[2] == 0 or roi[3] == 0:
            print("Skipped or invalid ROI selection.")
            continue

        x, y, w, h = roi
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pbar = tqdm(total=total_frames, desc="Extracting Non-Duplicate ROI Frames")
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Evaluate frame based on FRAME_STEP
            if frame_idx % FRAME_STEP == 0:
                # Crop ROI [y:y+h, x:x+w]
                crop = frame[int(y):int(y+h), int(x):int(x+w)]
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

                current_hash = calculate_dhash(gray_crop)

                # Deduplication Check
                is_duplicate = False
                for saved_hash in kept_hashes:
                    if hamming_distance(current_hash, saved_hash) <= HASH_DIFF_THRESHOLD:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    kept_hashes.append(current_hash)
                    out_name = f"trim_vf4_roi_frame_{total_saved:05d}.jpg"
                    cv2.imwrite(os.path.join(OUTPUT_FOLDER, out_name), crop)
                    total_saved += 1

            frame_idx += 1
            pbar.update(1)

        cap.release()
        pbar.close()

    print("\n" + "=" * 60)
    print(" EXTRACTION COMPLETE")
    print("=" * 60)
    print(f" Total Unique ROI Frames Extracted : {total_saved}")
    print(f" Saved Location                   : {OUTPUT_FOLDER}")
    print("=" * 60)

if __name__ == "__main__":
    process_videos()