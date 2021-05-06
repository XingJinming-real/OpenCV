import numpy as np
import cv2
import matplotlib.pyplot as plt

"""geometric ops"""


def HSVPractice():
    def empty(x):
        pass

    img = cv2.imread('../Resources/circle.png')
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.namedWindow('mask')
    cv2.namedWindow('imgNew')
    cv2.createTrackbar('lowerBound', 'mask', 0, 255, empty)
    cv2.createTrackbar('upperBound', 'mask', 20, 255, empty)
    while True:
        #  通过cv2.cvtColor(np.uint8([[[0,0,255]]],cv2.COLOR_BGR2HSV)求得red在hsv中的表示
        cv2.imshow('temp', img)
        lower = cv2.getTrackbarPos('lowerBound', 'mask')
        upper = cv2.getTrackbarPos('upperBound', 'mask')
        lowerBound = np.array([lower, 220, 220])
        upperBound = np.array([upper, 255, 255])
        mask = cv2.inRange(imgHsv, lowerBound, upperBound)  # 用来生成mask，在范围内为255，不在为0
        cv2.imshow('mask', mask)
        imgNew = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow("imgNew", imgNew)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break


def writeVideo():
    cap = cv2.VideoCapture(0)
    capWriter = cv2.VideoWriter('temp.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0,
                                (640, 480))  # *'XVID'表示('X','V','I','D')
    #  固定使用‘XVID'
    if not cap.isOpened():
        pass
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                pass
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('temp', frame)
                capWriter.write(frame)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
    cap.release()
    capWriter.release()


def GeometricTransformation():
    img = cv2.imread("../Resources/girl.jpg", 0)

    def scaling():
        imgBigger = cv2.resize(img, None, fx=2, fy=2)
        cv2.imshow("temp", imgBigger)
        imgSmaller = cv2.resize(img, None, fx=0.5, fy=0.5)
        cv2.imshow('tempSmall', imgSmaller)
        cv2.waitKey(0)

    def translation():
        rows, cols = img.shape
        imgShift = cv2.warpAffine(img, np.float32([[1, 0, -100], [0, 1, -50]]), (rows - 100, cols - 50))
        # Translation is the shifting of an object's location. If you know the shift in the (x,y) direction and let
        # it be (tx,ty), you can create the transformation matrix M as follows:
        #
        # M=[1 0 tx
        #    0 1 ty]
        # 注意tx，ty可以为正或为负,分别表示向右，下；左，上移动
        # 这样表示将图片(以左上角为锚点)水平txpixel，竖直typixel
        # 第三个参数为变换后图像的大小，如果和原来一样大，则空处补零，为黑
        cv2.imshow("ori", img)
        cv2.imshow('tempResult', imgShift)
        cv2.waitKey(0)

    def rotation():
        rows, cols = img.shape
        M = cv2.getRotationMatrix2D(((rows - 1) / 2, (cols - 1) / 2), -90, 0.7)
        imgNew = cv2.warpAffine(img, M, (rows, cols))
        cv2.imshow('temp', imgNew)
        cv2.waitKey(0)

    def affineTransformation():
        figure, ax = plt.subplots(1, 2)
        pts1 = np.array([[200, 20], [400, 400], [400, 0]], np.float32)
        # 注意pts一定要是float32型
        pts2 = np.array([[0, 0], [400, 400], [0, 400]], np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        imgNew = cv2.warpAffine(img, M, img.shape)
        cv2.imshow('temp', imgNew)
        cv2.imshow("ori", img)
        cv2.waitKey(0)

    def mouseEvent(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)

    def perspectiveTransformation():
        """同Affine(使用3个点求变换矩阵
        ,只不过要根据四个点求变换矩阵"""
        cv2.namedWindow('temp', cv2.WINDOW_AUTOSIZE)
        img = cv2.imread("../Resources/answerCard.png", 0)
        """184 244 lt
            631 250 rt
            658 829 rb
            169 830 lb"""
        # cv2.setMouseCallback('temp',mouseEvent)
        M = cv2.getPerspectiveTransform(np.float32([[184, 244], [196, 830],
                                                    [638, 829], [631, 250]]),
                                        np.float32([[0, 0], [0, 300],
                                                    [300, 300], [300, 0]]))
        imgNew = cv2.warpPerspective(img, M, (300, 300))
        cv2.imshow('temp', imgNew)
        cv2.waitKey(0)

    perspectiveTransformation()


if __name__ == '__main__':
    HSVPractice()
    # writeVedio()
    # GeometricTransformation()
    pass
