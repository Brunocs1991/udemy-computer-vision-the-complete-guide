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
img = cv2.imread("../data/people1.jpg")
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
