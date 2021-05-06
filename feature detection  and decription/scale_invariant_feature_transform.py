import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def SIFT():
    """harris can't deal with an img when it is scaled, so a scale invarient
    algorithm is proposed called SIFT"""
    """sift.detect() function finds the keypoint in the images. You can pass a mask if you want to search only a part of 
    image. Each keypoint is a special structure which has many attributes like its (x,y) coordinates, 
    size of the meaningful neighbourhood, angle which specifies its orientation, 
    response that specifies strength of keypoints etc. if you specifies cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS to it,
    it it will draw a circle with size of keypoint and it will even show its orientation """

    img = cv.imread('../Resources/corner.png')
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(imgGray, None)
    # result1 = cv.drawKeypoints(img, kp, img)
    result2 = cv.drawKeypoints(img, kp, img)
    # cv.imshow("result1", result1)
    cv.imshow("result2", result2)
    """now we wanna to get descriptors we can use sift.compute(imgGray,kp), or just use sift.detectAndCompute to get kp
    and des in one step like kp,des=cv.detectAndCompute(imgGray,None)"""
    cv.waitKey(0)
    # descriptor is a array of shape Number_of_Keypoints * 128
    """MATCHING WILL BE IN THE LAST TWO CHAPTERS"""


SIFT()
