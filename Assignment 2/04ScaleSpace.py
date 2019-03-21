'''
Created on Feb 11, 2014

@author: Yuon
'''
import scipy.ndimage as ndimage
import cv2
import numpy as np

def ConstructLoG(img,sigmas):
    LoG =  np.zeros((img.shape[0], img.shape[1], len(sigmas)), dtype=np.float32)
    kernel = np.array([[0, -1, 0, ],[-1, 4, -1],[0, -1, 0,]],dtype=np.float)
    for i in range(0, len(sigmas)-1):
        current_sigma = sigmas[i]
        img_tmp=ndimage.gaussian_filter(img,current_sigma)
        img_tmp=ndimage.convolve(img_tmp,kernel)
        img_tmp *= current_sigma*current_sigma
        LoG[:,:,i] = img_tmp
    
    return LoG    



img = cv2.imread("4Dots.png")
if img == None:
    print "Input image not loaded !!!"
    
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gray = np.array(img_gray,dtype=np.float)

LoG = ConstructLoG(img_gray,[2,4,6,8])
LoG -= np.min(LoG)
LoG /=np.max(LoG)

print LoG.shape
