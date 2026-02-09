import os
from PIL import Image
import numpy as np

DATA_DIR = "dataset/processed"

def trim_and_save():
    print(f"Checking {DATA_DIR}...")
    
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".png"):
                path = os.path.join(root, file)
                
                try:
                    img = Image.open(path).convert("RGBA")
                    
                    # Get the alpha channel (transparency)
                    alpha = np.array(img)[:, :, 3]
                    
                    # Find the bounding box of non-zero alpha
                    coords = np.argwhere(alpha > 0)
                    
                    if coords.size > 0:
                        y_min, x_min = coords.min(axis=0)
                        y_max, x_max = coords.max(axis=0)
                        
                        # Crop the image to the object
                        cropped = img.crop((x_min, y_min, x_max+1, y_max+1))
                        
                        # Save it back, overwriting the original
                        cropped.save(path)
                        print(f"Fixed: {file} (New size: {cropped.size})")
                    else:
                        print(f"Skipping empty image: {file}")
                        # Optional: os.remove(path) # Delete empty files
                        
                except Exception as e:
                    print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    trim_and_save()
