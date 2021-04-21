import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

"""threshold ops"""


def main():
    img = cv.imread("../Resources/girl.jpg", 0)

    def simpleThreshold():
        __, img_Binary = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        # >127=255，otherwise=0
        __, img_Trunc = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
        #  >127=127,otherwise maintain
        __, img_toZero = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
        #  >127 maintain,otherwise=0
        __, img_OTSU = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # 自动计算出阈值，其思想是：
        # 首先获得图像的histogram，找到一个值，使得该值对histogram中所有peak的方差最小
        img_AdaptiveMean = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                                cv.THRESH_BINARY, 11, 2)
        # 上述Binary的阈值是全局的，现在使用局部的threshold，
        # The threshold value is the mean of the neighbourhood area minus the constant C.
        img_AdaptiveGaussian = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv.THRESH_BINARY, 11, 2)
        # 同上
        figure, ax = plt.subplots(1, 6, figsize=(14, 8))
        titleList = ['binary', 'truc', 'toZero', "OTSU", 'AdaMean', "AdaGaussian"]
        imgList = [img_Binary, img_Trunc, img_toZero, img_OTSU, img_AdaptiveMean, img_AdaptiveGaussian]
        for i in range(6):
            ax[i].imshow(imgList[i], 'gray', vmin=0, vmax=255)
            ax[i].set_title(titleList[i])
        plt.suptitle('Threshold')
        plt.show()

    simpleThreshold()


if __name__ == "__main__":
    main()
