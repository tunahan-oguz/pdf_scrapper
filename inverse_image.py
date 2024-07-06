"""
Take the inverse of the image if most of the image is black!
"""
import cv2
import os
import numpy as np
image_root = r"dataset\images\14.pdf"
imgs = os.listdir(image_root)

for img_name in imgs:
    img_path = os.path.join(image_root, img_name)
    print(img_path)
    color_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    med = np.median(color_image)
    if med < 150:
        color_image = 255 - color_image
    cv2.imwrite(img_path, color_image)
