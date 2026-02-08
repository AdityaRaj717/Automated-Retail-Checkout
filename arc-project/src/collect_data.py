import cv2
import os
import time
from rembg import remove
from PIL import Image
import numpy as np

# --- CONFIGURATION ---
# CHANGE THIS NAME for every new item you scan! 
# Example: "maggi_ketchup", "hajmola", "tic_tac_toe"
ITEM_NAME = "moms_magic" 

# Directories
BASE_DIR = "dataset"
RAW_DIR = os.path.join(BASE_DIR, "raw", ITEM_NAME)
PROC_DIR = os.path.join(BASE_DIR, "processed", ITEM_NAME)

# Create folders if they don't exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

print(f"--- COLLECTING DATA FOR: {ITEM_NAME} ---")
print(f"Saving to: {PROC_DIR}")
print("Press 'SPACE' to capture | Press 'Q' to quit")

# Start Webcam (Try 0, 1, or 2 if 0 fails)
# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture("http://10.91.73.101:4747/video")

# Set resolution (Optional, helps with DroidCam quality)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame.")
        break

    # Show the live feed
    cv2.imshow('Data Collector', frame)

    key = cv2.waitKey(1) & 0xFF

    # Capture on SPACE
    if key == ord(' '):
        timestamp = int(time.time())
        
        # 1. Save Raw (Backup)
        raw_filename = f"{timestamp}.jpg"
        raw_path = os.path.join(RAW_DIR, raw_filename)
        cv2.imwrite(raw_path, frame)
        print(f"Captured: {raw_filename}")

        # 2. Remove Background (The Magic Step)
        print("Processing... ", end="", flush=True)
        
        try:
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Remove BG
            output = remove(pil_image)
            
            # Save as PNG (to keep transparency)
            proc_filename = f"{timestamp}.png"
            proc_path = os.path.join(PROC_DIR, proc_filename)
            output.save(proc_path)
            print("Done!")
        except Exception as e:
            print(f"\nError processing image: {e}")

    # Quit on Q
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
