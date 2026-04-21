# ==========================================
# INDIAN TRAFFIC SIGN
#
# Corrupted Image Removal - Technique Used: File Validation using OpenCV
# Duplicate Image Removal - Technique Used: MD5 Hash-Based Duplicate Detection
# Blur Detection (Outlier Removal) - Technique Used: Variance of Laplacian Method
# Image Resizing - Technique Used: Spatial Rescaling
# Normalization - Technique Used: Min-Max Normalization
# Label Encoding - Technique Used: Integer Encoding
# Dataset Splitting - Technique Used: Stratified Sampling
# ==========================================

import os
import cv2
import shutil
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# 1. DATASET PATH
# ==========================================

DATASET_PATH = r"C:\Users\gshiv\Desktop\Project Internship\Traffic Sign Project\Traffic Sign Datasets\Indian Traffic Sign\Images"
IMG_SIZE = 224

# ==========================================
# 2. CLASS ID → NAME MAPPING
# ==========================================

class_names = {
0:"Give way",
1:"No entry",
2:"One-way traffic",
3:"One-way traffic",
4:"No vehicles in both directions",
5:"No entry for cycles",
6:"No entry for goods vehicles",
7:"No entry for pedestrians",
8:"No entry for bullock carts",
9:"No entry for hand carts",
10:"No entry for motor vehicles",
11:"Height limit",
12:"Weight limit",
13:"Axle weight limit",
14:"Length limit",
15:"No left turn",
16:"No right turn",
17:"No overtaking",
18:"Maximum speed limit (90 km/h)",
19:"Maximum speed limit (110 km/h)",
20:"Horn prohibited",
21:"No parking",
22:"No stopping",
23:"Turn left",
24:"Turn right",
25:"Steep descent",
26:"Steep ascent",
27:"Narrow road",
28:"Narrow bridge",
29:"Unprotected quay",
30:"Road hump",
31:"Dip",
32:"Loose gravel",
33:"Falling rocks",
34:"Cattle",
35:"Crossroads",
36:"Side road junction",
37:"Side road junction",
38:"Oblique side road junction",
39:"Oblique side road junction",
40:"T-junction",
41:"Y-junction",
42:"Staggered side road junction",
43:"Staggered side road junction",
44:"Roundabout",
45:"Guarded level crossing ahead",
46:"Unguarded level crossing ahead",
47:"Level crossing countdown marker",
48:"Level crossing countdown marker",
49:"Level crossing countdown marker",
50:"Level crossing countdown marker",
51:"Parking",
52:"Bus stop",
53:"First aid post",
54:"Telephone",
55:"Filling station",
56:"Hotel",
57:"Restaurant",
58:"Refreshments"
}

# Save class names for later prediction use
pd.Series(class_names).to_csv("class_names.csv")

# ==========================================
# 3. LOAD FOLDERS
# ==========================================

classes = sorted(os.listdir(DATASET_PATH), key=lambda x: int(x))
print("Total Classes:", len(classes))

# ==========================================
# 4. CLEANING PROCESS
# ==========================================

def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < 100

hashes = {}
X = []
y = []

print("\nCleaning and loading dataset...")

for cls in classes:
    folder = os.path.join(DATASET_PATH, cls)
    label = int(cls)
    
    for img_name in os.listdir(folder):
        path = os.path.join(folder, img_name)
        
        # Remove corrupted
        img = cv2.imread(path)
        if img is None:
            os.remove(path)
            continue
        
        # Remove blurry
        if is_blurry(img):
            os.remove(path)
            continue
        
        # Remove duplicates
        with open(path, 'rb') as f:
            img_hash = hashlib.md5(f.read()).hexdigest()
        
        if img_hash in hashes:
            os.remove(path)
            continue
        else:
            hashes[img_hash] = path
        
        # Resize & Normalize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        
        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset Shape:", X.shape)
print("Labels Shape:", y.shape)

# ==========================================
# 5. SPLIT DATASET
# ==========================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("\nTrain:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# ==========================================
# 6. SAVE DATA
# ==========================================

np.save("Indian Traffic Sign/X_train.npy", X_train)
np.save("Indian Traffic Sign/X_val.npy", X_val)
np.save("Indian Traffic Sign/X_test.npy", X_test)

np.save("Indian Traffic Sign/y_train.npy", y_train)
np.save("Indian Traffic Sign/y_val.npy", y_val)
np.save("Indian Traffic Sign/y_test.npy", y_test)

print("\n🚦 CLEANING COMPLETED SUCCESSFULLY")