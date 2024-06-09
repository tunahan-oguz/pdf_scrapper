"""
Take the inverse of the image if most of the image is black!
"""
import cv2
import os

image_root = "/home/oguz/Desktop/BIL471/project/pdf_scrapper/dataset/images/4.pdf"
imgs = os.listdir(image_root)

for img_name in imgs:
    img_path = os.path.join(image_root, img_name)
    print(img_path)
    color_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if (color_image == 255).sum() >= (color_image == 0).sum(): continue
    color_image = 255 - color_image
    cv2.imwrite(img_path, color_image)
