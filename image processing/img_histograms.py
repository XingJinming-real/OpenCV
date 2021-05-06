import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

"""
this py file includes four sections
1.how to get and draw histogram
2.hist equalization(global) and local hist equalization (using clahe) 
3.two dimensions hist (using h and s of hsv)
4.backProject for object tracking (using color information) sliding the window"""
"""
we can use cv.calcHist or np.histogram() hist
"""
# plot hist graph
# img = cv.imread("../Resources/home.png")
# imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# histGray = cv.calcHist([imgGray], [0], None, [256], [0, 256])
# histNp, bins = np.histogram(imgGray.ravel(), 256, [0, 256])
# histBinCount = np.bincount(imgGray.ravel())
# fig, ax = plt.subplots(2, 2, figsize=(14, 14))
# for i, color in enumerate(('b', 'g', 'r')):
#     tempHist = cv.calcHist([img], [i], None, [256], [0, 256])
#     ax[0][0].plot(tempHist, color=color)
# ax[0][1].set_title('colorHist')
# ax[0][1].plot(histNp)
# ax[1][0].set_title('histNp')
# ax[1][0].set_title('histBinCount')
# ax[1][0].plot(histBinCount)


# hist equalization
# first we use standard histEqualization
# fig, ax = plt.subplots(2, 2, figsize=(14, 7))
# ax[0][0].imshow(imgGray, cmap='gray')
# ax[0][0].set_title('Ori')
# imgResult = cv.equalizeHist(imgGray)
# ax[0][1].imshow(imgResult, cmap='gray')
# ax[0][1].set_title('Standard')
# we first create clahe object
"""clahe short for contrast limit adaptive histogram equalization
@:parameter1 clipLimit: 如果bin之间的值相差大于clipLimit，则这个区域内的像素将会平均分配到其他bin中
@:parameter2 default is tileGridSize=(8,8) the small tile to compute
"""

# clahe = cv.createCLAHE(clipLimit=2)
# imgClahe = clahe.apply(imgGray)
# ax[1][0].imshow(imgClahe, cmap='gray')
# ax[1][0].set_title('Clahe')
# plt.show()

# 2D(two dimensions) histogram
"""first we convert rgb to hsv, then we have 2 parameters that are h(hue, 色调) and s(saturation, 饱和度)"""
# h[0,180),s[0,256),v[0,256)
# imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# hist = cv.calcHist([imgHSV], [0, 1], None, [180, 256], [0, 180, 0, 256])
# # or we simpily use hist,xbins,ybins=np.histogram2d(h.ravel(),s.ravel(),
# plt.imshow(hist, interpolation='nearest')
# plt.show()
"""histogram backProjection"""


#
# np.histogram2d(img)
def empty(x):
    pass


cv.namedWindow('aaa', cv.WINDOW_AUTOSIZE)
cv.createTrackbar('threshold', 'aaa', 50, 255, empty)
imgROI = cv.imread('../Resources/grass2.png')
imgROIHSV = cv.cvtColor(imgROI, cv.COLOR_BGR2HSV)
img = cv.imread('../Resources/messi2.png')
histROI = cv.calcHist([imgROIHSV], [0, 1], None, [180, 256], [0, 180, 0, 256])
histROI = cv.normalize(histROI, histROI, 0, 255, cv.NORM_MINMAX)
dst = cv.calcBackProject([img], [0, 1], histROI, [0, 180, 0, 256], 1)
dst = cv.filter2D(dst, -1, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)))
while True:
    threshold = cv.getTrackbarPos('threshold', 'aaa')
    __, tempThreshold = cv.threshold(dst, threshold, 255, cv.THRESH_BINARY)
    temp = cv.bitwise_and(img, img, mask=tempThreshold)
    cv.imshow("aaa", temp)
    cv.waitKey(22)
