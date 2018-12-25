import cv2
import numpy as np

def showImage(_img, sec=0):
    cv2.imshow("image", _img)
    cv2.waitKey(sec)

def draw_seam(img, seam, interactive=False):
    cv2.polylines(img, np.int32([np.asarray(seam)]), False, (0, 255, 0))
    cv2.imshow('seam', img)
    cv2.waitKey(1)

    if not interactive:
        cv2.destroyAllWindows()