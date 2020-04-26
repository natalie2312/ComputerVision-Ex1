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
import cv2
import numpy as np

from ex1_utils import LOAD_GRAY_SCALE, imReadAndConvert
from ex1_utils import *

def gammaDisplay(img_path: str, rep: int):

    def nothing(gamma):
        gamma = cv2.getTrackbarPos('gamma', 'image')
        img=0
        if (gamma != 0):
            img = pow(imgOrig, (1.0 / (gamma / 100)));
        cv2.imshow('image', img)
        k = cv2.waitKey(1)
    pass
    # Create a window
    if(rep==LOAD_GRAY_SCALE):
        img = imReadAndConvert(img_path,rep)
        imgOrig = img
    else:
        img=cv2.imread(img_path)/255
        imgOrig = img
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('gamma', 'image', 1, 200, nothing)
    nothing(0)
    # Wait until user press some key
    cv2.waitKey()
    pass

def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)

if __name__ == '__main__':
    main()
