import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

"""we use GarbCut algorithm to get foreground"""
"""briefly saying the algorithm is that we have a cost function which is 
defined by the sum of weights(the more probably the pixels are on the edge,
the lower weights they get, and we want to minimize the cost function."""


def foreGroundExtraction():
    """
    @:parameter mask:
                    cv.GC_BGD (GrabCut_BackGrounD
                    cv.GC_FGD
                    cv.GC_PR_BGD (GrabCut_PRobably_BackGrounD)
                    cv.GC_PR_FGD
                    or simply pass 0,1,2,3 to the image
                mask: manually choose, all pixels not in the rect will be considered as the bg
                rect: it is the coordinates of a rectangle (x,y,w,h)
                bdgModel: simply pass np.zeros((1,65),np.float64)
                fgdModel: the same with above
                iterCount: obviously
                mode: choose one from cv.GC_INIT_WITH_RECT or cv.GC_INIT_WITH_MASK
                :return None
    for grayScale or cimg
    during the iteration the input mask will be modified in each loop specifically:
    bgPixel=0,fgPixel=1,bgPR=2,fgPR=3
    check http://dl.acm.org/citation.cfm?id=1015720 for more information"""
    img = cv.imread("../Resources/messi.jpg")
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    fg = np.zeros((1, 65), np.float64)
    bg = np.zeros((1, 65), np.float64)
    # for codes' simplicity we define the rect is the full img, and may not get a perfect result compared to specifies
    # the true area.
    x, y, (w, h) = 0, 0, img.shape[:2]
    mask = np.zeros((w, h), np.uint8)
    rect = (0, 0, w, h)
    cv.grabCut(img, mask, rect, bg, fg, 5, cv.GC_INIT_WITH_RECT)
    # we set bgPixel=0,fgPixel=1,bgPR=0,fgPR=1
    # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # attention, bitwise_and is not proper here, for we use img(BGR) and bitwise is executed bit per bit
    # img = img * mask2[:, :, np.newaxis]  # np.newaxis is for boardCast
    # cv.imshow('result', img)
    # cv.waitKey(0)

    """improve"""
    """here we use paint tool to add boarders manually"""
    """if the result doesn't perfect, just paint more lines"""
    newMask = cv.imread('../Resources/messiAfterPaint.jpg', 0)
    mask[newMask == 255] = 1
    mask[newMask == 0] = 0
    cv.grabCut(img, mask, None, bg, fg, 5, cv.GC_INIT_WITH_MASK)
    newMask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype(np.uint8)
    img = img * newMask2[:, :, np.newaxis]
    cv.imshow("result", img)
    cv.waitKey(0)
    

foreGroundExtraction()
