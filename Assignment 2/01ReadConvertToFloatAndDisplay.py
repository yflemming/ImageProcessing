'''
Created on Feb 10, 2014

@author: Yuon
'''
import cv2
import numpy as np

img = cv2.imread("4Dots.png")
print img.shape
if img == None:
    print "Input image not loaded !!!"
    
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gray = np.array(img_gray,dtype=np.float)

print np.min(img_gray)
print np.max(img_gray)


img_gray -= np.min(img_gray)
img_gray /=np.max(img_gray)

print np.min(img_gray)
print np.max(img_gray)

cv2.imshow('image 1', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()