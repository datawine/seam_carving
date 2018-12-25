import cv2
import numpy as np

def gaussian_blur(img):
    return cv2.GaussianBlur(img, (3, 3), 0, 0)

def x_gradient(img):
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3,
                     scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

def y_gradient(img):
    return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3,
                     scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def energy(img):
    blurred = gaussian_blur(img)
    gray = grayscale(blurred)
    dx = x_gradient(gray)
    dy = y_gradient(gray)

    return cv2.add(np.absolute(dx), np.absolute(dy))

def getL2Gradient(single_channel):
    x_gradient = cv2.Scharr(single_channel, -1, 1, 0)
    y_gradient = cv2.Scharr(single_channel, -1, 0, 1)
    return np.absolute(x_gradient) + np.absolute(y_gradient)