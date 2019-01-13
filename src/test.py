import matplotlib.pyplot as plt
import numpy as np
import cv2
from calc_energy import *

from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from debug_module import *

'''
img = cv2.imread("../images/1.jpg")
b, g, r = cv2.split(img)
img = b

#img = cv2.imread("lena.jpg", 0)
result2 = np.zeros(img.shape, dtype=np.float16)
h, w = img.shape
subwin_size = 5
for y in xrange(subwin_size, h-subwin_size):
   for x in xrange(subwin_size, w-subwin_size):
       subwin = img[y-subwin_size:y+subwin_size, x-subwin_size:x+subwin_size]
       #hist = genHist(subwin)         # Generate histogram
       entropy = calcEntropy(subwin)    # Calculate entropy
       result2.itemset(y,x,entropy)

for j in range(img.shape[1]):
    print result2[0][j]
#showImage(result2)
'''

img = cv2.imread("../res/mask_delete100.jpg")
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j][0] == 0:
            print img[i][j]