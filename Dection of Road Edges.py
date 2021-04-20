import numpy as np
import cv2 as cv


def empty(x):
    pass


cap = cv.VideoCapture("Resources/01.avi")
cv.namedWindow('result', cv.WINDOW_AUTOSIZE)

"""创建滑动条，方便寻找合适参数"""
cv.createTrackbar('binaryThreshold', 'result', 200, 255, empty)
# 化为二值图，灰度值>200为255，其余情况为0
cv.createTrackbar('houghThreshold', 'result', 20, 255, empty)
# 该直线上的像素点个数>20则被认为是可能的目标直线，<20则直接舍弃该直线
cv.createTrackbar('minLength', 'result', 265, 400, empty)
# 直线的最短长度，<265直接舍弃(从图中我们可以得到，路沿长度均大于265)，>265为可能的目标直线
# 该参数可以一定程度上过滤掉一些非目标直线
cv.createTrackbar('maxLenGap', 'result', 40, 400, empty)
# 被认为是相同的一条直线之间的最大间隔，如果间隔>40则认为两个相邻的线段不属于同一条直线，否则属于同一条直线
# 该参数设置过大会导致一些本不是同一条直线上的直线段被强行认为是一条直线，导致一些误判直线
# 设置过小会遗漏掉一些本应该连起来的直线，对前述操作后图片的容错率下降，不能很好的对Canny检测后的图像进行补救操作
cv.createTrackbar('lowerBound', 'result', 60, 255, empty)
# Canny的下边界
cv.createTrackbar('upperBound', 'result', 150, 255, empty)
# Canny的上边界
# 此处设置下边界*2.5=上边界

while True:
    if not cap.isOpened():
        break
    ret, img = cap.read()
    if not ret:
        break

    """将显示时间的部分去掉"""
    imgResult = img.copy()
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ROI = imgGray.copy()
    rows, cols = imgGray.shape
    ROI[:60, :] = np.zeros((60, cols))

    """获取滑动条的值"""
    lowerBound = cv.getTrackbarPos('lowerBound', 'result')
    upperBound = cv.getTrackbarPos('upperBound', 'result')
    binaryThreshold = cv.getTrackbarPos('binaryThreshold', 'result')
    houghThreshold = cv.getTrackbarPos('houghThreshold', 'result')
    minLength = cv.getTrackbarPos('minLength', 'result')
    maxLenGap = cv.getTrackbarPos('maxLenGap', 'result')

    """将图像变为二值图，方便进行Canny和Hough"""
    __, ROI2 = cv.threshold(ROI, binaryThreshold, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    # 创建椭圆形卷积核
    ROI3 = cv.morphologyEx(ROI2, cv.MORPH_CLOSE, kernel)
    # 进行一次形态学闭操作，即先dilate再erode，以求增强边界效果，同时去除一些孤立的噪声白点和白色毛刺状噪声
    ROI4 = cv.morphologyEx(ROI3, cv.MORPH_DILATE, kernel)

    # 再进行一次dilate，将原本较为稀疏但处于目标直线上的点膨胀，增强Canny处理效果
    ROI5 = cv.Canny(ROI4, lowerBound, upperBound)
    cv.imshow("temp", ROI5)
    cv.waitKey(0)
    # 进行Canny边缘检测
    Lines = cv.HoughLinesP(ROI5, 1, np.pi / 180, houghThreshold,
                           minLineLength=minLength, maxLineGap=maxLenGap)
    # 使用HoughTransform进行直线检测
    # XingJinming2019284091
    if Lines is None:
        continue
        # 如果没有检测出，则处理下一帧
    for line in Lines:
        # 对每一个检测出的可能目标直线
        x1, y1, x2, y2 = line[0]
        """
                    np.abs(x1 - x2) <= 10 or np.abs(y1 - y2) <= 10:
                    前者表示直线接近垂直，后者表示直线接近水平，显然不是待求边界
                    (x2 - x1) / (y1 - y2) > 1 or (x2 - x1) / (y2 - y1) > 0.8:
                    我们从图中可以得知待求直线的斜率较小，上述判断去除了斜率较大的很可能不是目标的直线
                    (np.abs(x1 - x2) < 20 and np.abs(y1 - y2) > 150):
                    这里是对第一条的补充，表示如果直线接近垂直(水平变化较小，垂直变换较大)，则不是待求目标
                """
        if np.abs(x1 - x2) <= 10 or np.abs(y1 - y2) <= 10 or (x2 - x1) / (
                y1 - y2) > 1 or (x2 - x1) / (y2 - y1) > 0.8 or (
                np.abs(x1 - x2) < 20 and np.abs(y1 - y2) > 150):
            continue
        print("x1={},y1={}\nx2={},y2={}".format(x1, y1, x2, y2))
        cv.line(imgResult, (x1, y1), (x2, y2), (0, 0, 255), 3)
    print(20 * '*')
    cv.imshow("result", imgResult)
    cv.waitKey(20)
