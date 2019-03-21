'''
Created on Feb 3, 2014

@author: Ilan
'''
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
sigma = 3.0
N= int(4*sigma)

kernel = []
for i in range (-N, N):
    kernel.append(math.exp( -(i**2)/(2*(sigma**2))))

kernelAr = np.array(kernel)
total = sum(kernel)

normalKernel = kernelAr/total

arg = np.linspace(0,10*np.pi,100)
x = np.sin(arg)
x = x + 0.5 * rnd.randn(100)

y = np.zeros((x.shape[0]),dtype=np.float32)

for i in range(0, len(y) -1):
    y_i=0
    for j in range (-N, N):
        if((i-j)>=0) and ((i-j)<x.shape[0]):
            y_i += kernel[j + N]*x[i-j]
    y[i]=y_i
    
plt.subplot(211)
plt.plot(x)
plt.title('Input Noisy Signal')
plt.subplot(212)
plt.plot(y)
plt.title('Filtered Signal')
plt.show()