"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import time
from typing import List

import cv2
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    return 205946221


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    im = cv2.imread(filename)
    if len(im.shape) < 3 and representation == 2:
        im= cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    elif len(im.shape) == 3:
        if representation == 1:
            im= cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        elif representation == 2:
            im= cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    im = cv2.normalize(im.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return im



def imDisplay(filename: str, representation: int):
    np_im = imReadAndConvert(filename, representation)
    plt.imshow(np_im)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    if len(imgRGB.shape) == 3:
        yiq_ = np.array([[0.299, 0.587, 0.114],
                        [0.596, -0.275, -0.321],
                        [0.212, -0.523, 0.311]])
        imYI = np.dot(imgRGB, yiq_.T.copy())
        return imYI


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    if len(imgYIQ.shape) == 3:
        rgb_ = np.array([[1.00, 0.956, 0.623],
                        [1.0, -0.272, -0.648],
                        [1.0, -1.105, 0.705]])
        imRGB = np.dot(imgYIQ, rgb_.T.copy())
        return imRGB


def hsitogramEqualize(imOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    temp = imOrig.copy()
    if len(imOrig.shape) == 3:
        imOrig = cv2.normalize(imOrig, None, 0, 255, cv2.NORM_MINMAX)
        imOrig = np.ceil(imOrig)
        imOrig = imOrig.astype('uint8')
        imOrig = cv2.cvtColor(imOrig, cv2.COLOR_BGR2RGB)
        imOrig = cv2.normalize(imOrig.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        imyiq = transformRGB2YIQ(imOrig)
        imOrig = imyiq[:, :, 0]
    # nirmul
    imOrig = cv2.normalize(imOrig, None, 0, 255, cv2.NORM_MINMAX)
    imOrig = np.ceil(imOrig)
    imOrig = imOrig.astype('uint8')
    hist1, bins = np.histogram(imOrig.flatten(), 256, [0, 256])

    cdf = hist1.cumsum()
    cdf_normalized = cdf * hist1.max() / cdf.max()

    plt.plot(cdf_normalized, color='b')
    plt.hist(imOrig.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('CDF', 'Histogram'), loc='upper left')
    plt.show()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    imOrig = imOrig.astype('uint8')
    img2 = cdf[imOrig]

    hist2, bins = np.histogram(img2.flatten(), 256, [0, 256])
    cdf2 = hist2.cumsum()
    cdf_normalized2 = cdf2 * hist2.max() / cdf2.max()

    plt.plot(cdf_normalized2, color='b')
    plt.hist(img2.flatten(), 256, [0, 256], color='g')
    plt.xlim([0, 256])
    plt.legend(('CDF', 'Histogram'), loc='upper left')
    plt.show()
    if len(temp.shape) == 3:
        img2 = cv2.normalize(img2.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        imyiq[:, :, 0] = img2
        img2 = transformYIQ2RGB(imyiq)
        img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
        img2 = np.ceil(img2)
        img2 = img2.astype('uint8')
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    return img2, hist1, hist2



# gets an 0-255 image and extract the Y from YIQ converted, in 0-1 array.
def extractYchannelFromBGR(imOrig: np.ndarray) -> np.ndarray:
    yChannel = np.copy(imOrig)
    yChannel = cv2.cvtColor(yChannel, cv2.COLOR_BGR2RGB)
    yChannel = cv2.normalize(yChannel.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    yChannel = transformRGB2YIQ(yChannel)  # yChannel is now YIQ
    yChannel = yChannel[:, :, 0]
    return yChannel


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int):
    imQ = np.copy(imOrig)
    imQ2 = np.copy(imOrig)
    indexBestQuant = []   # might delete
    segments = np.array([0, 255])
    # getting Z:
    for i in range(1, nQuant):
        segments = np.insert(segments, i, int((256 / nQuant) * i))

    plt.title('Equalized image histogram with CDF')
    plt.hist(imQ.flatten(), 256, [0, 255], color='r')
    plt.xlim([0, 255])
    plt.legend(('cdf - EQUALIZED', 'histogram - EQUALIZED'), loc='upper right')
    plt.show()
    print(imQ2)
    print(segments)
    # between segments
    for i in range(0, len(segments) - 1):
        histOrig, bins = np.histogram(imQ.flatten(), int(256 / nQuant), [segments[i], segments[i + 1] - 1])
        print(histOrig)  # delete
        cdf = histOrig.cumsum()
        print('cdf: ' + str(cdf))  # delete
        histProb = np.array([])
        avg = 0
        for j in range(0, len(histOrig)):
            # histProb = np.insert(histProb, j, histOrig[j]/cdf[-1])
            avg = avg + (histOrig[j] * (histOrig[j] / cdf[-1]))
        print(avg)
        idxOfValueNearAVG = find_nearest(histOrig, avg)
        actualColorNearAvg = idxOfValueNearAVG + i * int(256 / nQuant)
        print(actualColorNearAvg)
        indexBestQuant.append(actualColorNearAvg)
        imQ2[(imQ2 < segments[i+1]) & (imQ2 >= segments[i])] = actualColorNearAvg

        plt.hist(imQ.flatten(), int(255 / nQuant), [segments[i], segments[i + 1]], color='r')
        plt.xlim([0, 255])
        plt.show()
    print(indexBestQuant)
    cv2.imshow('Quantized image', imQ2)
    pass


def histEqDemo(img_path: str, rep: int):
    img = imReadAndConvert(img_path, rep)
    imgeq, histOrg, histEq = hsitogramEqualize(img)

    # Display cumsum
    cumsum = np.cumsum(histOrg)
    cumsumEq = np.cumsum(histEq)
    plt.gray()
    plt.plot(range(256), cumsum, 'r')
    plt.plot(range(256), cumsumEq, 'g')

    # Display the images
    plt.figure()
    plt.imshow(img)

    plt.figure()
    plt.imshow(imgeq)
    plt.show()