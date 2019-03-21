'''
Created on Mar 4, 2014

@author: Yuon
'''

import cv2
import numpy.linalg as lin
import numpy as np
import matplotlib.pyplot as plt


def bestMatches(des1,des2):
    best_matches = []
    for i in range(0, des1.shape[0]):
        des1dex = 0
        des2dex = 0
        dij = 1000
        for j in range(0, des2.shape[0]):
            dist = lin.norm((des2[j] - des1[i]))
            
            if dist < dij:
                dij = dist
                des1dex = i
                des2dex = j
                
        best_matches.append([des1dex, des2dex, dij])
    return best_matches
        

img1 = cv2.imread("wes1.jpg")
img2 = cv2.imread("wes2.jpg")

img_gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img_gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()

[kp1,des1] = sift.detectAndCompute(img_gray1, None)
[kp2,des2] = sift.detectAndCompute(img_gray2, None)

best_matches = bestMatches(des1, des2)

sorted_matches = sorted(best_matches, key = lambda x:x[2])

dists = np.zeros(len(sorted_matches))
for k in range(0, len(sorted_matches)-1):
    dists[k] = sorted_matches[k][2] 
 
[histogram, edges] = np.histogram(dists, 20, (0,450))
 
bin_width = edges[1]- edges[0]
 
plt.bar(edges[0:-1], histogram, bin_width)
plt.show()

