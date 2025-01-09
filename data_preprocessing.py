import cv2
import numpy as np
import os

def load_data(test):
    images = []
    labels = []
    label_map = {label: idx for idx, label in enumerate(os.listdir(test))}
    for label in os.listdir(test):
        for img_file in os.listdir(os.path.join(test, label)):
            img = cv2.imread(os.path.join(test, label, img_file))
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label_map[label])
    
    print(f"Loaded {len(images)} images and {len(labels)} labels.")
    return np.array(images, dtype=np.float32), np.array(labels, dtype=int)  # Use int instead of np.int
