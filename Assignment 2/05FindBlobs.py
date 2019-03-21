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

def FindMaxima3D(LoG,scales):
    maxima = {}
    max_value = np.max(LoG)
    for i in range(1, LoG.shape[0]-2):
        for j in range(1, LoG.shape[1]-2):
            for k in range(1, LoG.shape[2]-2):
                if LoG[i,j,k]>LoG[i,j,k-1] and LoG[i,j,k]>LoG[i,j,k+1] and LoG[i,j,k]>LoG[i,j-1,k] and LoG[i,j,k]>LoG[i,j+1,k] and LoG[i,j,k]>LoG[i-1,j,k] and LoG[i,j,k]>LoG[i+1,j,k] and LoG[i,j,k]>0.5*max_value:
                    maxima[(j,i)]=scales[k]
    
    return maxima


img = cv2.imread("4Dots.png")
if img == None:
    print "Input image not loaded !!!"
    
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gray = np.array(img_gray,dtype=np.float)

LoG = ConstructLoG(img_gray, [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8])
maxima = FindMaxima3D(LoG,[1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8])

print 'Found ', len(maxima), 'maxima'
klist = maxima.keys()
for i in range(0, len(maxima)):
    k=klist[i]
    print 'maxima at position -', k, 'maxima scale = ', maxima[k]

for k in maxima:
    cv2.circle(img,k,int(maxima[k]),(0,0,255),1)
    
cv2.imshow('image_and_circles_scale' , img)
cv2.waitKey(0)
cv2.destroyAllWindows()