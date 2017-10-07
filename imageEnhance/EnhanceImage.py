import cv2
import numpy as np

# function to determine the bounding box of the text. To crop the image
def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    print ymin,ymax,xmin,xmax
    return ymin,ymax,xmin,xmax
    #return img[ymin:ymax+1, xmin:xmax+1]


#dilates the text
def dilate(image):
    
    # Load an color image in grayscale
    img = cv2.imread(image,0)
    _, im_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV )
    
    kernel = np.ones((3,3),np.uint8)
    dilate = cv2.dilate(im_thresh,kernel,iterations = 1)
    _, dil_image = cv2.threshold(dilate, 145, 255, cv2.THRESH_BINARY_INV)

    return dil_image


#determines the orientation and rotates the image
def orientImage(image):
    
    img = image.copy()
    
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(img)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
     
    combinedContour = np.vstack(cnt for cnt in hulls)
    combinedContour = np.array(combinedContour)
    
    hull = cv2.convexHull(combinedContour)
    
    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    
    angle = cv2.minAreaRect(hull)[-1]
    print angle 
    
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle 
    if angle < -45:
        angle = (90 + angle)
     
    # otherwise, just take the angle
    else: 
        angle = angle 
    
    print angle    
    # rotate the image to deskew it
    (h, w) = img.shape[:2] 
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
    return rotated

        
image = "fisdom.jpg"
dilated = dilate(image)   

rotated = orientImage(dilated)   

_, im_thresh = cv2.threshold(rotated, 127, 255, cv2.THRESH_BINARY_INV )

ymin,ymax,xmin,xmax = bbox2(im_thresh)  

text =  rotated[ymin:ymax+1, xmin:xmax+1]  
text_gray = cv2.cvtColor(text,cv2.COLOR_GRAY2BGR)

#Code for blending the colour
for i in range(len(text_gray)):
    for j in range(len(text_gray[0])):
        if (text_gray[i][j][0] < 127 and text_gray[i][j][1] < 127 and text_gray[i][j][2] < 127):
            text_gray[i][j] = [128,0,0]
            
            
cv2.imshow("text_gray", text_gray)
cv2.waitKey(0)               
