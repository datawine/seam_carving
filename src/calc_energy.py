import cv2
import numpy as np
import pySaliencyMap

def getGrayL1Gradient(img):
    blurred = cv2.GaussianBlur(img, (3, 3), 0, 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    return cv2.add(np.absolute(dx), np.absolute(dy))

def getRGBL1Gradient(img):
    b, g, r = cv2.split(img)
    b_energy = np.absolute(cv2.Sobel(b, cv2.CV_64F, 1, 0)) + np.absolute(cv2.Sobel(b, cv2.CV_64F, 0, 1))
    g_energy = np.absolute(cv2.Sobel(g, cv2.CV_64F, 1, 0)) + np.absolute(cv2.Sobel(g, cv2.CV_64F, 0, 1))
    r_energy = np.absolute(cv2.Sobel(r, cv2.CV_64F, 1, 0)) + np.absolute(cv2.Sobel(b, cv2.CV_64F, 0, 1))
    return cv2.add(cv2.add(b_energy, g_energy), r_energy)

def getRGBL2Gradient(img):
    b, g, r = cv2.split(img)
    b_energy = np.absolute(cv2.Sobel(b, cv2.CV_64F, 2, 0)) + np.absolute(cv2.Sobel(b, cv2.CV_64F, 0, 2))
    g_energy = np.absolute(cv2.Sobel(g, cv2.CV_64F, 2, 0)) + np.absolute(cv2.Sobel(g, cv2.CV_64F, 0, 2))
    r_energy = np.absolute(cv2.Sobel(r, cv2.CV_64F, 2, 0)) + np.absolute(cv2.Sobel(b, cv2.CV_64F, 0, 2))
    return cv2.add(cv2.add(b_energy, g_energy), r_energy)

def getRGBLaplacian(img):
    b, g, r = cv2.split(img)
    b_energy = np.absolute(cv2.Laplacian(b, cv2.CV_64F))
    g_energy = np.absolute(cv2.Laplacian(g, cv2.CV_64F))
    r_energy = np.absolute(cv2.Laplacian(r, cv2.CV_64F))
    return cv2.add(cv2.add(b_energy, g_energy), r_energy)

def getSaliency(img):
    imgsize = img.shape
    img_width  = imgsize[1]
    img_height = imgsize[0]
    sm = pySaliencyMap.pySaliencyMap(img_width, img_height)
    # computation
    saliency_map = sm.SMGetSM(img)
    return saliency_map

def calcEntropy(img):
    #hist,_ = np.histogram(img, np.arange(0, 256), normed=True)
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    hist = hist.ravel()/hist.sum()
    #logs = np.nan_to_num(np.log2(hist))
    logs = np.log2(hist+0.00001)
    #hist_loghist = hist * logs
    entropy = -1 * (hist*logs).sum()
    return entropy