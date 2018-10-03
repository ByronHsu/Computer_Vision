import numpy as np
import cv2
new = cv2.imread('0.4-0.0-0.6-2a-f.png')
# new = cv2.imread('1.0-0.0-0.0-2a-f.png')
old = cv2.imread('testdata/2a.png')
old = old[6:-7, 6:-7, :]
a = np.linalg.norm(old - new)
print(a)