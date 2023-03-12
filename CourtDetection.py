import numpy as np
import matplotlib.pyplot as plt
from cv2 import VideoCapture, cvtColor, Canny, line, imshow, waitKey, destroyAllWindows, COLOR_BGR2GRAY, HoughLines, threshold, THRESH_BINARY, dilate, floodFill
from TraceHeader import videoFile, checkPath

video = VideoCapture(videoFile)
checkPath(videoFile)
width = int(video.get(3))
height = int(video.get(4))

while video.isOpened():
    ret, frame = video.read()
    if frame is None:
        break
    
    gry = cvtColor(frame, COLOR_BGR2GRAY)

    bw = threshold(gry, 160, 255, THRESH_BINARY)[1]
    dilation = dilate(bw, np.ones((5, 5), np.uint8), iterations=1)
    nonRectArea = dilation.copy()
    floodFill(nonRectArea, np.zeros((height+2, width+2), np.uint8), (width//2, height//2), 0)
    dilation[np.where(nonRectArea == 255)] = 0
    canny = Canny(dilation, 100, 200)
    
    ret, videoFrame = video.read()
    if videoFrame is None:
        break
    hLines = HoughLines(canny, 2, np.pi/180, 300)
    for Line in hLines:
        for rho,theta in Line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 2000*(-b))
            y1 = int(y0 + 2000*(a))
            x2 = int(x0 - 2000*(-b))
            y2 = int(y0 - 2000*(a))
            line(videoFrame, (x1,y1), (x2,y2), (0, 0, 255), 2)
            
    imshow("Frame", videoFrame)
    if waitKey(1000000) == ord("q"):
        break
    
video.release()
destroyAllWindows()