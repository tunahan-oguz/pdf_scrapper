"""
Take the inverse of the image if most of the image is black!
"""
import cv2
import os
import numpy as np
image_root = "/home/oguz/Downloads/13.pdf"
imgs = os.listdir(image_root)

for img_name in imgs:
    img_path = os.path.join(image_root, img_name)
    print(img_path)
    color_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    med = np.median(color_image)
    if med < 150:
        color_image = 255 - color_image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    B, G, R = cv2.split(color_image)
    B = clahe.apply(B)
    G = clahe.apply(G)
    R = clahe.apply(R)
    cv2.imwrite(img_path, cv2.merge((B, G, R)))
