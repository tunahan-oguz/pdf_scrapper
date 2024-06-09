import pandas as pd
import cv2
import numpy as np

csv_path = "dataset/descriptions/image_descriptions7.pdf.csv"
df = pd.read_csv(csv_path)
df = df.dropna()

for index, row in df.iterrows():
    path, desc = row
    desc = desc.replace("\n", " ")
    image = cv2.imread(path)
    image = cv2.resize(image, (720, 720))
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    cv2.putText(image, text=desc, org=(15, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=255)
    cv2.imshow("image", image)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    cv2.destroyAllWindows()
cv2.destroyAllWindows()
