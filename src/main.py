from debug_module import *
from calc_energy import *
from seam_carving import *
import cv2
import numpy as np

for i in range(1, 7):
    if i == 2:
        img = cv2.imread("../images/2.png")
    else:
        img = cv2.imread("../images/" + str(i) + ".jpg")
    print img.shape

    seam_instance = SeamCarving(img)
    output = seam_instance.seam_delete(int(img.shape[0] * 0.7), int(img.shape[1] * 0.7))
    if i == 2:
        cv2.imwrite("../res/rgbL1/delete_2.png", output)    
    else:
        cv2.imwrite("../res/rgbL1/delete_" + str(i) + ".jpg", output)
    output = seam_instance.seam_insert(int(img.shape[0] * 1.3), int(img.shape[1] * 1.3))
    if i == 2:
        cv2.imwrite("../res/rgbL1/insert_2.png", output)    
    else:
        cv2.imwrite("../res/rgbL1/insert_" + str(i) + ".jpg", output)
