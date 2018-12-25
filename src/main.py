from debug_module import *
from calc_energy import *
from seam_carving import *
import cv2
import numpy as np

def calcEnergy(_img):
    b, g, r = cv2.split(_img)
    b_energy = getL2Gradient(b)
    g_energy = getL2Gradient(g)
    r_energy = getL2Gradient(r)
    return b_energy + g_energy + r_energy

for i in range(3, 7):
    img = cv2.imread("../images/" + str(i) + ".jpg")
    print img.shape

    seam_instance = SeamCarving(img)
    output = seam_instance.seam_delete(int(img.shape[0] * 0.7), int(img.shape[1] * 0.7))
    cv2.imwrite("../res/" + str(i) + ".jpg", output)