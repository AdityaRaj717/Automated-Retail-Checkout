import torch
import cv2
import numpy as np
import socketio
import base64
import time
import os
import sqlite3
from rembg import remove, new_session
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_URL = "http://localhost:3000"
MODEL_PATH = os.path.join(BASE_DIR, "retail_model.pth")
CLASS_FILE = os.path.join(BASE_DIR, "classes.txt")
DB_FILE = os.path.join(BASE_DIR, "products.db")
CAMERA_SOURCE = "http://10.91.73.101:4747/video" 

CONFIDENCE_THRESHOLD = 50 
MIN_OBJECT_AREA = 500

# Global State
cart = []
product_db = {} # Cache for DB
scan_trigger = False
sio = socketio.Client()

# --- DATABASE HELPERS ---
def load_product_db():
    global product_db
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT * FROM products")
        rows = c.fetchall()
        # Create a dict: {'key_name': {'name': 'Display Name', 'price': 10}}
        product_db = {r[0]: {'name': r[1], 'price': r[2]} for r in rows}
        conn.close()
        print(f"[INFO] Loaded {len(product_db)} products from Database.")
    except Exception as e:
        print(f"[ERROR] Database Error: {e}. Run setup_db.py first!")

# --- SOCKET HANDLERS ---
@sio.on('connect')
def on_connect():
    print("[INFO] Connected to Node.js Server")

@sio.on('execute_scan')
def on_execute_scan(data=None):
    global scan_trigger
    print("[CMD] Received Web Trigger -> Scanning...")
    scan_trigger = True

@sio.on('cart_action_trigger')
def on_cart_action(data):
    global cart
    action_type = data.get('type')
    item_id = data.get('id')
    
    print(f"[ACTION] {action_type} on item {item_id}")
    
    if action_type == 'clear':
        cart = []
        
    elif action_type == 'remove':
        cart = [item for item in cart if item['id'] != item_id]
        
    elif action_type == 'increment':
        for item in cart:
            if item['id'] == item_id:
                item['qty'] += 1
                item['total_price'] = item['unit_price'] * item['qty']
                
    elif action_type == 'decrement':
        for item in cart:
            if item['id'] == item_id:
                if item['qty'] > 1:
                    item['qty'] -= 1
                    item['total_price'] = item['unit_price'] * item['qty']
    
    # Send updated cart back to UI immediately
    sio.emit('cart_update', cart)

# --- ML SETUP ---
def load_system():
    with open(CLASS_FILE, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    
    session = new_session()
    return model, class_names, device, session

def main():
    global scan_trigger, cart
    
    # Init
    load_product_db()
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
    print("--- SYSTEM READY ---")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Stream Video
        stream_frame = cv2.resize(frame, (640, 360))
        _, buffer = cv2.imencode('.jpg', stream_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        b64_image = base64.b64encode(buffer).decode('utf-8')
        sio.emit('processed_frame', {'image': b64_image})

        key = cv2.waitKey(1) & 0xFF
        
        # Detection Logic
        if scan_trigger or key == ord(' '):
            scan_trigger = False
            print("\n--- PROCESSING IMAGE ---")
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            pil_no_bg = remove(pil_img, session=session)
            
            alpha_mask = np.array(pil_no_bg)[:, :, 3]
            contours, _ = cv2.findContours(alpha_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            found_items_count = 0
            
            for cnt in contours:
                if cv2.contourArea(cnt) < MIN_OBJECT_AREA: continue 
                
                # Crop & Predict
                x, y, w, h = cv2.boundingRect(cnt)
                item_crop = pil_no_bg.crop((x, y, x+w, y+h))
                
                max_dim = max(w, h)
                bg = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
                offset_x = (max_dim - w) // 2
                offset_y = (max_dim - h) // 2
                bg.paste(item_crop, (offset_x, offset_y), mask=item_crop.split()[3])
                
                img_t = transform(bg).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(img_t)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    conf, idx = torch.max(probs, 1)
                
                conf_val = conf.item() * 100
                key_name = classes[idx.item()]
                
                if conf_val > CONFIDENCE_THRESHOLD:
                    # Get details from DB (fallback to defaults if missing)
                    db_item = product_db.get(key_name, {'name': key_name, 'price': 0})
                    display_name = db_item['name']
                    unit_price = db_item['price']
                    
                    # Update Cart Logic
                    existing_item = next((item for item in cart if item['key_name'] == key_name), None)
                    
                    if existing_item:
                        existing_item['qty'] += 1
                        existing_item['total_price'] = existing_item['qty'] * unit_price
                        print(f"  + Incremented: {display_name}")
                    else:
                        new_item = {
                            'id': int(time.time() * 1000) + x, # Unique ID
                            'key_name': key_name,
                            'display_name': display_name,
                            'unit_price': unit_price,
                            'qty': 1,
                            'total_price': unit_price
                        }
                        cart.append(new_item)
                        print(f"  + New: {display_name}")
                    
                    found_items_count += 1

            if found_items_count > 0:
                sio.emit('cart_update', cart)
                print(f"Cart sync sent.")
            else:
                print("No items found.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    sio.disconnect()

if __name__ == "__main__":
    main()
