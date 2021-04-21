import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def main():
    """
    kernel is given by users
    you can get rect,ellipse,cross size kernel from cv.getStructuringElement(shape,size)
    shape is from cv.MORPH_RECT,cv.MORPH_Ellipse,cv.MORPH_CROSS; size is tuple,
    dilate: the central value will be 1 as long as there exits 1 in the area of kernel
    erode: the central value will be 0 as long as there exits 0 in the area of kernel

    cv.open is erode followed by dilation, while the close is the opposite one
    iteration=2(open/close) is the same with erode->erode->dilate->dilate

    topHat: the difference between img and opening img   "img - open
    blackHat: the difference between img and closing img  "img - close
    gradient: get the edges
    :return: None
    """
    plt.ion()
    # kernel = np.ones((3, 3))
    img = cv.imread('../Resources/morphological.png')
    kernelRect = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    imgDilate = cv.dilate(img, kernelRect)
    imgErosion = cv.erode(img, kernelRect)
    kernelEllipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    kernelCross = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    imgOpen = cv.morphologyEx(img, cv.MORPH_OPEN, kernelRect)
    imgClose = cv.morphologyEx(img, cv.MORPH_CLOSE, kernelRect)
    """open is often for small white noise in black background"""
    """close is often for small black holes-like noise in white foreground"""
    imgTopHat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernelRect)
    imgBlackHat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernelRect)
    imgGradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernelRect)
    figure, ax = plt.subplots(1, 2, figsize=(14, 7))
    imgList = [imgDilate, imgErosion, imgGradient, imgOpen, imgClose, imgBlackHat, imgTopHat]
    for i in range(7):
        ax[0].imshow(img)
        ax[0].set_title('Ori')
        ax[1].imshow(imgList[i])
        ax[1].set_title(i)
        plt.pause(4)
    plt.ioff()


if __name__ == '__main__':
    main()
