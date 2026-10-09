import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FOLDER = "Unzipped_frames"      # Folder containing your raw ROI images
OUTPUT_FOLDER = "output_frames/filtered_unique" # Folder where clean unique images will be saved
HASH_THRESHOLD = 6                                 # Lower = stricter (drops more images), Higher = keeps more variation

def calculate_dhash(image, hash_size=8):
    """Generates a 64-bit difference hash for an image."""
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()

def hamming_distance(hash1, hash2):
    """Calculates bit differences between two hashes."""
    return np.count_nonzero(hash1 != hash2)

def filter_dataset():
    if not os.path.exists(INPUT_FOLDER):
        print(f"[ERROR] Input folder '{INPUT_FOLDER}' does not exist.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Get sorted image files
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_exts)])
    
    total_images = len(image_files)
    if total_images == 0:
        print("[ERROR] No images found in the input folder.")
        return

    print(f"Loaded {total_images} raw ROI frames from '{INPUT_FOLDER}'. Filtering duplicates...\n")

    kept_hashes = []
    kept_count = 0
    dropped_count = 0

    for img_name in tqdm(image_files, desc="Deduplicating Frames"):
        img_path = os.path.join(INPUT_FOLDER, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue

        current_hash = calculate_dhash(img)

        # Check if current frame is too similar to any previously kept frame
        is_duplicate = False
        for saved_hash in kept_hashes:
            if hamming_distance(current_hash, saved_hash) <= HASH_THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_hashes.append(current_hash)
            src_path = os.path.join(INPUT_FOLDER, img_name)
            dst_path = os.path.join(OUTPUT_FOLDER, img_name)
            shutil.copy2(src_path, dst_path)
            kept_count += 1
        else:
            dropped_count += 1

    print("\n" + "=" * 45)
    print(" DEDUPLICATION RESULTS")
    print("=" * 45)
    print(f" Total Raw Images Processed : {total_images}")
    print(f" Duplicate Frames Removed  : {dropped_count} ({dropped_count/total_images*100:.1f}%)")
    print(f" Final Clean Unique Images  : {kept_count}")
    print(f" Saved To                   : {OUTPUT_FOLDER}")
    print("=" * 45)

if __name__ == "__main__":
    filter_dataset()