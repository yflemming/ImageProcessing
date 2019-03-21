'''
Created on Feb 26, 2014

@author: Yuon
'''

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage

sigma = 10.0

x = np.zeros((100,100),dtype=np.float32)
x[50][50]=1


Hii = ndimage.gaussian_filter(x, sigma, (2,0))
Hij = ndimage.gaussian_filter(x, sigma, (1,1))
Hjj = ndimage.gaussian_filter(x, sigma, (0,2))

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


