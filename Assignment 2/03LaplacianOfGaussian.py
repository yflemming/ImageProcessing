'''
Created on Feb 11, 2014

@author: Yuon
'''
import scipy.ndimage as ndimage
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

sigma = 6.0

img_gray=ndimage.gaussian_filter(img_gray,sigma)

kernel = np.array([[0, -1, 0, ],[-1, 4, -1],[0, -1, 0,]],dtype=np.float)

img_gray=ndimage.convolve(img_gray,kernel)
img_gray *= sigma*sigma

img_gray -= np.min(img_gray)
img_gray /=np.max(img_gray)

print np.min(img_gray)
print np.max(img_gray)

cv2.imshow('image 3', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()