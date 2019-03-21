'''
Created on Feb 27, 2014

@author: Yuon
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import cv2
import numpy.linalg as lin

sigmas = [0.01, 0.05, 0.1, 0.3, 0.5]

img = cv2.imread("Angiography_2D.png")
