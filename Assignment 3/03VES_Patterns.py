'''
Created on Feb 26, 2014

@author: Yuon
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import cv2
import numpy.linalg as lin
import math

sigmas = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
img = cv2.imread("PatternsBlack.png")
    
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gray = np.array(img_gray,dtype=np.float)

img_gray -= np.min(img_gray)
img_gray /= np.max(img_gray)

for sigma in sigmas:
    Hii = ndimage.gaussian_filter(img_gray, sigma, (2,0))
    Hij = ndimage.gaussian_filter(img_gray, sigma, (1,1))
    Hjj = ndimage.gaussian_filter(img_gray, sigma, (0,2))
    
    Hii *= sigma**2
    Hij *= sigma**2
    Hjj *= sigma**2
    
    for i in range(len(img_gray)):
        for j in range(len(img_gray[0])):
            currentPixArr = [ [Hii[i][j], Hij[i][j]],
                             [Hij[i][j], Hjj[i][j]] ]
            eigVals = lin.eigvals(currentPixArr)
            absVals = abs(eigVals)
            lambda1 = 0.0
            lambda2 = 0.0
        
            if absVals[0] >= absVals[1]:
                lambda1 = eigVals[1]
                lambda2 = eigVals[0]
            else:
                lambda1 = eigVals[0]
                lambda2 = eigVals[1]
            ves = []
            currentVES = 0.0
        
            if lambda2 > 0:
                currentVES = math.exp(-((lambda1/lambda2)/(2.0*0.5))**2) * (1 - math.exp(-((lambda1**2 + lambda2**2)/(2*((125*125))))))

            
            ves.append(currentVES)
            print(ves)
            
            img_gray[i][j] = np.max(ves)


plt.subplot(121)
plt.imshow(img)
plt.title("Original Image")

plt.subplot(122)
plt.imshow(img_gray, cmap="Greys_r")
plt.title("Vesselness Enhancement")
plt.show()
