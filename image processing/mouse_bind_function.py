import numpy as np
import cv2


def empty(x):
    pass


def draw(event, x, y, flags, param):
    global bX, bY, drawF, c, radius, color
    if event == cv2.EVENT_LBUTTONDOWN:
        bX = x
        bY = y
        drawF = True
        print('down')
    if event == cv2.EVENT_LBUTTONUP:
        drawF = False
        print('up')
    if drawF:
        print("c is {}".format(c))
        if c:
            cv2.circle(img, (x, y), radius, (0, 0, color), -1)
        else:
            cv2.rectangle(img, (bX, bY), (x, y), (0, color, 0), -1)
    pass


bX = 0
bY = 0
drawF = False
c = True
radius = 0
color = 0
if __name__ == "__main__":
    img = cv2.imread('../Resources/paper section.jpg', 0)
    cv2.namedWindow("img")
    cv2.createTrackbar('color', 'img', 0, 255, empty)
    cv2.createTrackbar('radius', 'img', 2, 10, empty)
    while True:
        cv2.setMouseCallback('img', draw)
        cv2.imshow("img", img)
        radius, color = cv2.getTrackbarPos('radius', 'img'), cv2.getTrackbarPos('color', 'img')
        k = cv2.waitKey(1) & 0xFF  # & is bitwise_and 操作
        if k == ord('m'):
            c = not c
        elif k == 27:
            break
