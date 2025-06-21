# Computer vision the complete guide
# Opencv

# %%
import sys
import os

sys.path.append(os.path.abspath('..'))

# %%
import cv2  # OpenCV
from utils.image_utils import show_image

# %%
img = cv2.imread("../data/images/people1.jpg")
# %%
img.shape

# %%
show_image(img, "People 1")

# %%
img = cv2.resize(img, (800, 600))

# %%
img.shape

# %%
show_image(img, "People 1 resized")
#%%
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img.shape
show_image(gray_img, "People 1 Gray")

 # %%
 ## face detection
face_detector = cv2.CascadeClassifier("../data/cascade/haarcascade_frontalface_default.xml")

# %%
detections = face_detector.detectMultiScale(gray_img)
detections

# %%
len(detections)

# %%
for (x, y, w, h) in detections:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
show_image(img, "People 1 with detections")
# %%
# Parameters haarcascades
img = cv2.imread("../data/images/people1.jpg")
img = cv2.resize(img, (800, 600))
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detections = face_detector.detectMultiScale(gray_img, scaleFactor=1.09)
for (x, y, w, h) in detections:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
show_image(img, "People 1 with detections (scaleFactor=1.09)")

#%%
img = cv2.imread("../data/images/people2.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detections = face_detector.detectMultiScale(gray_img, 
                                            scaleFactor=1.2,
                                            minNeighbors=3,
                                            minSize=(32, 32), maxSize=(100,100))
for (x, y, w, h) in detections:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
show_image(img, "People 2 with detections (scaleFactor=1.09)")

#%%
