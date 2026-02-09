import socketio
import cv2
import base64
import torch
import numpy as np
from rembg import remove, new_session
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import time
import os

import os 

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

SERVER_URL = "http://localhost:3000"
MODEL_PATH = os.path.join(PROJECT_ROOT, "retail_model.pth")
CLASS_FILE = os.path.join(PROJECT_ROOT, "classes.txt")

CAMERA_SOURCE = "http://10.91.73.101:4747/video" 
CONFIDENCE_THRESHOLD = 70

PRICES = {
    '50-50_maska_chaska': 30, 'aim_matchstick': 2, 'farmley_panchmeva': 375,
    'hajmola': 65, 'maggi_ketchup': 120, 'moms_magic': 9,
    'monaco': 35, 'tic_tac_toe': 20
}

# --- SOCKET SETUP ---
sio = socketio.Client()

@sio.event
def connect():
    print("[INFO] Connected to Node.js Server")

@sio.event
def disconnect():
    print("[INFO] Disconnected from Server")

def load_system():
    with open(CLASS_FILE, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    
    session = new_session()
    return model, class_names, device, session

def main():
    try:
        sio.connect(SERVER_URL)
    except Exception as e:
        print(f"[ERROR] Could not connect to Node server: {e}")
        return

    model, classes, device, session = load_system()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cart = []
    
    print("--- ARC SYSTEM LIVE ---")
    
    last_processed = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Logic Loop (Detect on SPACE bar press logic simulated here?)
        # For web, let's detect every 30 frames OR when motion stops?
        # For now, let's keep it manual scan triggered by keyboard on the terminal,
        # OR just stream the video and scan continuously (heavy).
        
        # OPTIMIZATION: Resize frame for streaming to reduce lag
        stream_frame = cv2.resize(frame, (640, 360))
        _, buffer = cv2.imencode('.jpg', stream_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        b64_image = base64.b64encode(buffer).decode('utf-8')
        
        # Emit Live Feed
        sio.emit('processed_frame', {'image': b64_image, 'last_scanned': None})

        # --- KEYBOARD TRIGGER (Terminal Side) ---
        # You still press SPACE in the terminal window to trigger a scan
        # We use a dummy cv2 window just to capture keypresses if needed, 
        # or we can auto-scan. Let's stick to keypress for accuracy.
        cv2.imshow("Controller (Press SPACE)", cv2.resize(frame, (300, 200)))
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            print("Scanning...")
            # ... (Insert your Contour/Prediction Logic Here) ...
            # [Copy the contour/prediction logic from previous turn]
            # When item found:
            # item_name = "maggi_ketchup" 
            # conf = 98.5
            
            # --- MOCK PREDICTION FOR TEST ---
            # (Replace this with real inference code)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            pil_no_bg = remove(pil_img, session=session)
            # ... process crop ...
            # Assume we found Maggi:
            
            # sio.emit('cart_update', cart)
            # --------------------------------
            
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    sio.disconnect()

if __name__ == "__main__":
    main()
