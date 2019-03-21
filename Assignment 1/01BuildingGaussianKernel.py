'''
Created on Feb 3, 2014

@author: Yuon
'''
import math
import matplotlib.pyplot as plt
import numpy as np

sigma = 3.0
N= int(4*sigma)

kernel = []
for i in range (-N, N):
    kernel.append(math.exp( -(i**2)/(2*(sigma**2))))

plt.plot(kernel)
plt.show()