import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

plt.ion()
figure, ax = plt.subplots(1, 2, figsize=(17, 7))
img = cv.imread("Resources/girl.jpg")
imgPyramid = [img]
for i in range(5):
    imgPyramid.append(cv.pyrDown(imgPyramid[i]))

imgPyramidL = [imgPyramid[5]]
for i in range(5, 0, -1):
    imgUp = cv.pyrUp(imgPyramid[i])
    if imgPyramid[i - 1].shape[1] != imgUp.shape[1]:
        imgUp = imgUp[:, :-1, :]
    if imgPyramid[i - 1].shape[0] != imgUp.shape[0]:
        imgUp = imgUp[:-1, :, :]
    imgPyramidL.append(cv.subtract(imgPyramid[i - 1], imgUp))
reconstruct = [imgPyramidL[0]]
for i in range(5):
    imgUp = cv.pyrUp(reconstruct[i])
    if imgUp.shape[0] > imgPyramidL[i + 1].shape[0]:
        imgUp = imgUp[:-1, :, :]
    if imgUp.shape[1] > imgPyramidL[i + 1].shape[1]:
        imgUp = imgUp[:, :-1, :]
    reconstruct.append(cv.add(imgUp, imgPyramidL[i + 1]))
for i in range(6):
    ax[0].imshow(imgPyramid[5 - i])
    ax[0].set_title('imgPyramid')
    ax[1].imshow(reconstruct[i])
    ax[1].set_title('Reconstruct')
    plt.pause(2)
cv.waitKey(0)
plt.pause(100)
plt.ioff()
plt.cla()
plt.close()
