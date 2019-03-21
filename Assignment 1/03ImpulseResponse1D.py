'''
Created on Feb 3, 2014

@author: Ilan
'''
import math
import matplotlib.pyplot as plt
import numpy as np
sigma = 3.0
N= int(4*sigma)

kernel = []
for i in range (-N, N):
    kernel.append(math.exp( -(i**2)/(2*(sigma**2))))

kernelAr = np.array(kernel)
total = sum(kernel)

normalKernel = kernelAr/total

x = np.zeros((100),dtype=np.float32)
x[50]=1

y = np.zeros((x.shape[0]),dtype=np.float32)

for i in range(0, len(y) -1):
    y_i=0
    for j in range (-N, N):
        if((i-j)>=0) and ((i-j)<x.shape[0]):
            y_i += kernel[j + N]*x[i-j]
    y[i]=y_i
    
plt.plot(y)
plt.show()