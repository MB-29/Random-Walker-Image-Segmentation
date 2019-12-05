import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage
import cv2

image_path = 'input/chien.jpg'
img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)

sx = cv2.Sobel(img, cv2.CV_64F,1,0)
sy = cv2.Sobel(img,cv2.CV_64F,0,1)
laplacian = cv2.Laplacian(img, cv2.CV_32F)

# plt.subplot(2,2,1)
# plt.imshow(sx,cmap='gray')
# plt.subplot(2,2,3)
# plt.imshow(sx,cmap='gray')
# plt.subplot(2,2,2)
# plt.imshow(sy, cmap='gray')
# plt.subplot(2,2,4)
# plt.imshow(sy, cmap='gray')
plt.imshow(laplacian)
plt.show()
