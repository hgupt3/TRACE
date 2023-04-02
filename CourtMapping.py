from skimage.transform import ProjectiveTransform
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

frameHeight = 800
frameWidth = 1400
ratio = (1097/2377)
courtHeight = frameHeight * 0.5
courtWidth = courtHeight * ratio
yOffset = (frameHeight - courtHeight) / 2
xOffset = (frameWidth - courtWidth) / 2

# def CourtMap(bottom_left, top_left, top_right, bottom_right, data):
#     t = ProjectiveTransform()
#     src = np.asarray([bottom_left, top_left, top_right, bottom_right])
#     dst = np.asarray([[0, 0], [0, 1], [1, 1], [1, 0]])
#     if not t.estimate(src, dst): raise Exception("estimate failed")
#     data = np.array(data)
#     data_transformed = t(data)
#     return data_transformed

def CourtMap(bottom_left, top_left, top_right, bottom_right, img):
    pts1 = np.float32([[top_left, top_right, bottom_left, bottom_right]])
    pts2 = np.float32([[0+xOffset,0+yOffset],[courtWidth+xOffset,0+yOffset],[0+xOffset,courtHeight+yOffset],[courtWidth+xOffset,courtHeight+yOffset]])
    M = cv.getPerspectiveTransform(pts1,pts2)
    dst = cv.warpPerspective(img,M,(frameWidth,frameHeight))
    return dst
    # plt.subplot(121),plt.imshow(img),plt.title('Input')
    # plt.subplot(122),plt.imshow(dst),plt.title('Output')
    # plt.show()

