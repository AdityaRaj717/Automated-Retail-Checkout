import torch
import cv2
import numpy as np
import socketio
import base64
import time
import os
from rembg import remove, new_session
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_URL = "http://localhost:3000"
MODEL_PATH = os.path.join(BASE_DIR, "retail_model.pth")
CLASS_FILE = os.path.join(BASE_DIR, "classes.txt")
CAMERA_SOURCE = "http://10.91.73.101:4747/video" 

# LOWERED THRESHOLD TO FIX MISSED DETECTIONS
CONFIDENCE_THRESHOLD = 50 

PRICES = {
    '50-50_maska_chaska': 30, 'aim_matchstick': 2, 'farmley_panchmeva': 375,
    'hajmola': 65, 'maggi_ketchup': 15, 'moms_magic': 10,
    'monaco': 35, 'tic_tac_toe': 20
}

sio = socketio.Client()
scan_trigger = False # Flag to control scanning

@sio.on('connect')
def on_connect():
    print("[INFO] Connected to Node.js Server")

@sio.on('execute_scan')
def on_execute_scan(data=None):
    global scan_trigger
    print("[CMD] Received Web Trigger -> Scanning...")
    scan_trigger = True

def load_system():
    with open(CLASS_FILE, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- UPDATED ARCHITECTURE TO MATCH TRAIN.PY ---
    # Load EfficientNet-B0 structure (no weights needed, we load ours next)
    model = models.efficientnet_b0(weights=None)
    
    # Recreate the classifier layer exactly as it was in training
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(class_names))
    # ---------------------------------------------

    # Load your trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    
    session = new_session()
    return model, class_names, device, session

def main():
    global scan_trigger
    try:
        sio.connect(SERVER_URL)
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")

    model, classes, device, session = load_system()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cart = []
    
    print(f"--- SYSTEM READY (Threshold: {CONFIDENCE_THRESHOLD}%) ---")
    print("Use the WEB INTERFACE (Spacebar) to scan items.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Stream Frame to Web (Reduced size for speed)
        stream_frame = cv2.resize(frame, (640, 360))
        _, buffer = cv2.imencode('.jpg', stream_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        b64_image = base64.b64encode(buffer).decode('utf-8')
        sio.emit('processed_frame', {'image': b64_image})

        # 2. Check for Trigger (Spacebar from Web OR 'c' locally)
        key = cv2.waitKey(1) & 0xFF
        
        if scan_trigger or key == ord(' '):
            scan_trigger = False # Reset flag
            print("\n--- PROCESSING SCAN ---")
            
            # Use original high-res frame for detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            
            # Remove Background
            pil_no_bg = remove(pil_img, session=session)
            
            # Find Contours on Alpha Channel
            alpha_mask = np.array(pil_no_bg)[:, :, 3]
            contours, _ = cv2.findContours(alpha_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            found_something = False
            
            for cnt in contours:
                if cv2.contourArea(cnt) < 2000: continue 
                
                x, y, w, h = cv2.boundingRect(cnt)
                item_crop = pil_no_bg.crop((x, y, x+w, y+h))
                
                # Create Training-Like Image (Black BG)
                bg = Image.new('RGB', item_crop.size, (0, 0, 0))
                bg.paste(item_crop, mask=item_crop.split()[3])
                
                # Predict
                img_t = transform(bg).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(img_t)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    conf, idx = torch.max(probs, 1)
                
                conf_val = conf.item() * 100
                item_name = classes[idx.item()]
                
                if conf_val > CONFIDENCE_THRESHOLD:
                    base_price = PRICES.get(item_name, 0)
                    
                    # Check if item exists in cart
                    existing_item = next((item for item in cart if item['name'] == item_name), None)
                    
                    if existing_item:
                        # UPDATE existing item
                        existing_item['qty'] += 1
                        existing_item['price'] = existing_item['qty'] * base_price
                        print(f"  + Updated: {item_name} (Qty: {existing_item['qty']})")
                    else:
                        new_item = {
                            'name': item_name, 
                            'price': base_price,
                            'qty': 1,
                            'id': int(time.time() * 1000) + x 
                        }
                        cart.append(new_item)
                        print(f"  + Added: {item_name}")
                    
                    found_something = True
                else:
                    print(f"  - Ignored: {item_name} ({conf_val:.1f}%) - Too unsure")

            if found_something:
                sio.emit('cart_update', cart)
                print(f"Sent {len(cart)} items to Web.")
            else:
                print("No confident items found.")

        elif key == ord('c'):
            cart = []
            sio.emit('cart_update', cart)
            print("Cart Cleared.")
            
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    sio.disconnect()

if __name__ == "__main__":
    main()
