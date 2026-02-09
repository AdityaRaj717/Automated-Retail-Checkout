# Installation Instructions

#### Install node modules and python packages before proceeding with the steps below 

1. Clone the repo
```git clone https://github.com/AdityaRaj717/Automated-Retail-Checkout.git```
2. Open 3 terminals
3. Run ```node index.js``` in the server directory
4. Run ```npm run dev``` in the client directory
5. Run ```python web_inference.py``` in the src directory
6. Go to http://localhost:5173

---

# Automated Retail Checkout System

### **1. Project Title & Objective**

* **Project Name:** ARC (Automated Retail Checkout) System
* **Objective:** To build a computer-vision-based "cashier-less" checkout system (similar to Amazon Go technology) that identifies products placed on a counter without barcodes, using Deep Learning.

### **2. System Architecture (The "Three-Tier" Stack)**

The project uses a hybrid architecture to combine high-performance Python AI with a modern React UI.

* **Tier 1: The Brain (Python & PyTorch)**
* **Role:** Handles Camera input, Image Processing, AI Inference, and Database Management.
* **Key Libraries:** `torch` (MobileNetV2), `cv2` (OpenCV), `rembg` (Background Removal), `sqlite3`.


* **Tier 2: The Bridge (Node.js & Socket.io)**
* **Role:** Acts as a real-time relay. It connects the "Frontend" (Browser) to the "Backend" (Python Script) which might be running on a separate device (like a Jetson Nano or local server).


* **Tier 3: The Interface (React + Tailwind)**
* **Role:** Displays the live camera feed, the generated bill, and allows user interaction (Add/Remove/Checkout).



---

### **3. Work Done & Features Implemented**

#### **Phase 1: Computer Vision & Model Training**

We went through several iterations to achieve high accuracy with a small dataset.

* **Data Strategy:** collected ~40 transparent images per class (8 classes: `maggi`, `good_day`, `matchsticks`, etc.).
* **The "Zoom Mismatch" Fix:** We realized the model was training on "tiny objects in a large frame" but predicting on "cropped large objects." We wrote `fix_dataset.py` to auto-crop training data, massively boosting accuracy.
* **Architecture Choice:** We tested **EfficientNet-B0** but reverted to **MobileNetV2**.
* *Reason:* MobileNetV2 is less prone to overfitting on small datasets (40 images/class) and runs faster on CPU/Edge devices.


* **Augmentation Strategy:** Implemented "Online Augmentation" in `train.py`.
* We used `rembg` to separate the object.
* We generate **random solid background colors** during training. This forces the AI to look at the *product packaging* (texture/text) rather than the shape/silhouette.



#### **Phase 2: The Inference Engine (`web_inference.py`)**

This is the core script running the show.

* **Multi-Object Detection:** It uses OpenCV Contours to find multiple items in one frame. It loops through them, crops them individually, and sends them to the AI model.
* **Small Object Logic:** Lowered detection thresholds (`Area > 500px`) to ensure small items like "Aim Matchsticks" are detected alongside large items like "Farmley".
* **Smart Cart Logic:**
* If an item is scanned again, it **increments the quantity** (`Qty: 2`) instead of creating a duplicate row.
* It calculates `Total Price` dynamically.



#### **Phase 3: Database Integration**

* **SQLite Integration:** Moved away from hardcoded Python dictionaries.
* **`products.db`:** A local database stores product names and prices. This allows the system to work offline and makes price updates easy without changing code.

#### **Phase 4: Full-Stack Interactivity**

* **Browser-Based Control:** We removed the dependency on the Python window. You can now press **Spacebar** on the React website to trigger the Python camera.
* **Two-Way Sync:**
* *Python -> React:* Sends Live Video Feed (Base64) and Cart Updates.
* *React -> Python:* Sends User Actions (Increment Qty, Decrement Qty, Remove Item, Clear Cart).



---

### **4. Challenges Solved**

| Challenge | Solution |
| --- | --- |
| **Model Overfitting** | Switched to MobileNetV2 and implemented heavy color jitter/random backgrounds. |
| **"Tiny Dot" Detection** | Created `fix_dataset.py` to crop training images to the bounding box. |
| **Bad UI Rendering** | Fixed Tailwind CSS v4 configuration in Vite. |
| **Separate Quantity Rows** | Updated Python logic to check `if item in cart` before appending. |
| **Dual Camera Feeds** | Optimized the pipeline to stream the detecting frame from Python to React, so the user sees exactly what the AI sees. |

### **5. Current Status**

* **Accuracy:** High (>90%) for the 8 trained classes.
* **Performance:** Real-time detections on standard hardware.
* **UI/UX:** Modern Dark Mode interface with Glassmorphism. Fully interactive cart.
* **Readiness:** The system is functionally complete for a Capstone demonstration.

### **Next Possible Steps (Optional)**

* **Checkout Feature:** Generate a PDF receipt when "Checkout" is clicked.
* **Payment Gateway:** Integrate a dummy QR code generator (UPI) for the total amount.
