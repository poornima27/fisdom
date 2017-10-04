import cv2
import numpy as np

# Load an color image in grayscale
img = cv2.imread('fisdom.jpg',0)
ret, im_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV )
cv2.imshow("image",im_thresh)
cv2.waitKey(0)

kernel = np.ones((5,5),np.uint8)
dilate = cv2.dilate(im_thresh,kernel,iterations = 1)
cv2.imshow("dialte",dilate)
cv2.waitKey(0)
# cv2.imwrite("result1.jpg",im_thresh_dil)
