import os
import cv2

def ensure_dir(directory_path):
    """Creates directory if it doesn't exist."""
    os.makedirs(directory_path, exist_ok=True)


def is_blurry(image, threshold=100.0):
    """
    Checks if an image is blurry using the Variance of Laplacian method.
    Returns True if blurry, False if sharp.
    """
    if image is None or image.size == 0:
        return True

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def select_roi(video_path):
    """
    Opens a resizable GUI window to let you drag and drop a box around desks/workstations.
    Returns tuple: (x, y, w, h).
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Error: Could not read frame from video to select ROI.")
        return None

    print("\n[INFO] Drag a box over the desk/workstation area and press ENTER or SPACE.")
    print("[INFO] Press 'c' to cancel selection.")

    # Create a resizable window for high-res videos
    window_name = "Select Desk ROI (Press ENTER/SPACE to confirm)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    roi = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)

    # Clean up GUI windows properly
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return roi