import torch
import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

# --- CONFIG ---
MODEL_PATH = "retail_model.pth"
CLASS_FILE = "classes.txt"
CAMERA_SOURCE = "http://10.91.73.101:4747/video"  # DroidCam
CONFIDENCE_THRESHOLD = 70  # Only bill if very sure

PRICES = {
    '50-50_maska_chaska': 10, 'aim_matchstick': 2, 'farmley_panchmeva': 500,
    'hajmola': 50, 'maggi_ketchup': 120, 'moms_magic': 20,
    'monaco': 10, 'tic_tac_toe': 150
}

def load_system():
    # Load Classes
    with open(CLASS_FILE, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load Rembg
    session = new_session()
    return model, class_names, device, session

def get_predictions(model, img_tensor, device, classes):
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)
        return classes[idx.item()], conf.item() * 100

def main():
    model, classes, device, session = load_system()
    
    # Preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cart = []
    
    print("--- MULTI-ITEM CHECKOUT READY ---")
    print("Rules: Items must NOT touch each other.")
    print("Press SPACE to Scan | C to Clear | Q to Quit")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        
        # UI Overlay
        cv2.putText(display_frame, f"Cart Size: {len(cart)}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Multi-Checkout", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            print("\nScanning Frame...")
            
            # 1. Remove Background (Get transparent PIL)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            pil_no_bg = remove(pil_img, session=session)
            
            # 2. Convert to CV2 Format to find Contours
            # Extract Alpha channel (the transparency mask)
            alpha_mask = np.array(pil_no_bg)[:, :, 3]
            
            # Find Contours (The "Blobs")
            contours, _ = cv2.findContours(alpha_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            print(f"Found {len(contours)} potential items...")
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 2000: continue  # Ignore small noise/dust
                
                # Get Bounding Box of this specific item
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Crop just this item from the original frame
                item_crop = pil_no_bg.crop((x, y, x+w, y+h))
                
                # Paste onto Black Square (to match training)
                bg = Image.new('RGB', item_crop.size, (0, 0, 0))
                bg.paste(item_crop, mask=item_crop.split()[3])
                
                # Predict
                img_t = transform(bg).unsqueeze(0).to(device)
                name, conf = get_predictions(model, img_t, device, classes)
                
                if conf > CONFIDENCE_THRESHOLD:
                    price = PRICES.get(name, 0)
                    cart.append({'name': name, 'price': price})
                    print(f"  + Added: {name} (Rs.{price}) [{conf:.1f}%]")
                else:
                    print(f"  - Ignored: {name} (Low Confidence: {conf:.1f}%)")

            total_bill = sum(item['price'] for item in cart)
            print(f"TOTAL BILL: Rs. {total_bill}")

        elif key == ord('c'):
            cart = []
            print("Cart Cleared.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
