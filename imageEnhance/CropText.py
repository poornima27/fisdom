import cv2
import numpy as np


def crop(file_name):
    
    mser = cv2.MSER_create()
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()
    rectImg = img.copy()
    regions, _ = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
   
    combinedContour = np.vstack(cnt for cnt in hulls)
    combinedContour = np.array(combinedContour)
   
    hull = cv2.convexHull(combinedContour)
    cv2.drawContours(vis,[hull],-1,(0,255,0),2)            
    cv2.imshow('img', vis)
    cv2.waitKey(0)
    
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(rectImg,[box],-1,(0,0,255),2)
    cv2.imshow('rectImg', rectImg)
    cv2.waitKey(0)
    
    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    cv2.drawContours(mask,[box],-1,(255,255,255),-1)   
   
    text_only = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('text_only', text_only)
    cv2.waitKey(0)
    
    res = np.hstack((img,vis))
    res1 = np.hstack((rectImg,text_only))
     
    result = np.vstack((res,res1))
    cv2.imshow('result', result)
    cv2.waitKey(0)
    
    cv2.imwrite("data1_crop.jpg",result)

file_name = 'data1.png'
crop(file_name)  