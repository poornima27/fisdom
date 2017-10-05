import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('fisdom.jpg',1)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, im_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV )
# cv2.imshow("image",im_thresh)
# cv2.waitKey(0)

mser = cv2.MSER_create()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
vis = img.copy()
rectImg = img.copy()
regions, _ = mser.detectRegions(gray)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
 
combinedContour = np.vstack(cnt for cnt in hulls)
combinedContour = np.array(combinedContour)

hull = cv2.convexHull(combinedContour)
cv2.drawContours(vis,[hull],-1,(0,255,0),1)            
cv2.imshow('img', vis)
cv2.waitKey(0)


# grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates
#coords = np.column_stack(np.where(im_thresh > 0))
angle = cv2.minAreaRect(hull)[-1]
print angle 
# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle 
if angle < -45:
    angle = (90 + angle)
 
# otherwise, just take the inverse of the angle to make
# it positive
else: 
    angle = angle 

print angle    
# rotate the image to deskew it
(h, w) = img.shape[:2] 
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(img, M, (w, h),
    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    
# draw the correction angle on the image so we can validate it
cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
# show the output image
print("[INFO] angle: {:.3f}".format(angle))
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)
cv2.imwrite("fisdom_result.jpg",rotated)
    
