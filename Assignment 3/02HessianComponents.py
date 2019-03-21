'''
Created on Feb 26, 2014

@author: Yuon
'''

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import cv2

sigma = 3.0

img = cv2.imread("PatternsBlack.png")
print img.shape
if img == None:
    print "Input image not loaded !!!"
    
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gray = np.array(img_gray,dtype=np.float)

img_gray -= np.min(img_gray)
img_gray /= np.max(img_gray)

Hii = ndimage.gaussian_filter(img_gray, sigma, (2,0))
Hij = ndimage.gaussian_filter(img_gray, sigma, (1,1))
Hjj = ndimage.gaussian_filter(img_gray, sigma, (0,2))

Hii *= sigma**2
Hij *= sigma**2
Hjj *= sigma**2

plt.subplot(321)
plt.plot(Hii)
plt.subplot(322)
plt.imshow(Hii)

plt.subplot(323)
plt.plot(Hij)
plt.subplot(324)
plt.imshow(Hij)

plt.subplot(325)
plt.plot(Hjj)
plt.subplot(326)
plt.imshow(Hjj)
plt.show()

