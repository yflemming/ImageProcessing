'''
Created on Mar 9, 2014

@author: Yuon
'''
import matplotlib.pyplot as plt
import cv2
import numpy.linalg as lin

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

img_rgb1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img_rgb2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

img_gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img_gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()

[kp1,des1] = sift.detectAndCompute(img_gray1, None)
[kp2,des2] = sift.detectAndCompute(img_gray2, None)

best_matches = bestMatches(des1, des2)

sorted_matches = sorted(best_matches, key = lambda x:x[2])
  
plt.subplot(211)
plt.imshow(img_rgb1)
for i in range(0, 10):
    plt.plot(kp1[sorted_matches[i][0]].pt[0], kp1[sorted_matches[i][0]].pt[1], 'rd')
plt.subplot(212)
plt.imshow(img_rgb2)
for i in range(0, 10):
    plt.plot(kp2[sorted_matches[i][1]].pt[0], kp2[sorted_matches[i][1]].pt[1], 'rd')
plt.show()
