import torch
import cv2
import numpy as np
from rembg import remove, new_session  # Import new_session
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

# --- CONFIG ---
MODEL_PATH = "retail_model.pth"
CLASS_FILE = "classes.txt"
# Use your DroidCam URL directly to be safe!
CAMERA_SOURCE = "http://10.91.73.101:4747/video" 

PRICES = {
    '50-50_maska_chaska': 10,
    'aim_matchstick': 2,
    'farmley_panchmeva': 500,
    'hajmola': 50,
    'maggi_ketchup': 120,
    'moms_magic': 20,
    'monaco': 10,
    'tic_tac_toe': 150
}
# --------------

def load_system():
    print("[INFO] Loading system...")
    
    # 1. Load Classes
    try:
        with open(CLASS_FILE, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"[ERROR] {CLASS_FILE} not found! Did you run train.py?")
        exit()

    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"[ERROR] {MODEL_PATH} not found! Did you run train.py?")
        exit()
        
    model = model.to(device)
    model.eval()
    
    # 3. Initialize Rembg Session (THE SPEED FIX)
    # This loads the background removal model into GPU memory ONCE.
    rembg_session = new_session() 
    
    print("[INFO] System Ready!")
    return model, class_names, device, rembg_session

def preprocess_image(cv2_frame, session):
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    
    # Remove BG using the pre-loaded session
    pil_no_bg = remove(pil_img, session=session)
    
    # Paste onto Black Background
    bg = Image.new('RGB', pil_no_bg.size, (0, 0, 0))
    # Safety check: ensure image actually has alpha channel
    if pil_no_bg.mode == 'RGBA':
        bg.paste(pil_no_bg, mask=pil_no_bg.split()[3])
    else:
        bg.paste(pil_no_bg)
    
    return bg

def main():
    model, classes, device, rembg_session = load_system()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"[INFO] Connecting to Camera: {CAMERA_SOURCE}")
    cap = cv2.VideoCapture(CAMERA_SOURCE) 
    
    if not cap.isOpened():
        print(f"[ERROR] Failed to open camera. Check DroidCam!")
        return

    cart_total = 0
    last_item = "Ready"
    
    print("\n--- RETAIL CHECKOUT SYSTEM LIVE ---")
    print("Press 'SPACE' to Scan | 'C' to Clear | 'Q' to Quit")

    while True:
        ret, frame = cap.read()
        if not ret: 
            print("Lost camera feed...")
            break

        # Interface
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (10, 10), (400, 130), (0, 0, 0), -1)
        cv2.putText(display_frame, f"TOTAL: Rs. {cart_total}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Last: {last_item}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Checkout System', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            cart_total = 0
            last_item = "Cleared"
            
        elif key == ord(' '):
            print("Scanning...", end="", flush=True)
            
            try:
                # 1. Preprocess (Fast Mode)
                clean_pil_img = preprocess_image(frame, rembg_session)
                
                # 2. Predict
                input_tensor = transform(clean_pil_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probs, 1)
                    confidence_pct = confidence.item() * 100

                item_name = classes[predicted_idx.item()]
                
                # 3. Logic
                if confidence_pct > 60:
                    price = PRICES.get(item_name, 0)
                    cart_total += price
                    last_item = f"{item_name} (Rs.{price})"
                    print(f" Detected: {item_name} ({confidence_pct:.1f}%)")
                else:
                    last_item = "Unknown Item"
                    print(f" Low Confidence ({confidence_pct:.1f}%)")
                    
            except Exception as e:
                print(f" [ERROR] Scan failed: {e}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
