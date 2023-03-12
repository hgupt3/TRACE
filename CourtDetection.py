import numpy as np
import matplotlib.pyplot as plt
from cv2 import VideoCapture, cvtColor, Canny, line, imshow, waitKey, destroyAllWindows, COLOR_BGR2GRAY, HoughLines
from cv2 import threshold, THRESH_BINARY, dilate, floodFill, circle
from TraceHeader import videoFile, checkPath, determinant

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
    
    hLines = HoughLines(canny, 2, np.pi/180, 350)
    lineEndpoints = []
    for hLine in hLines:
        for rho,theta in hLine:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + width*(-b))
            y1 = int(y0 + width*(a))
            x2 = int(x0 - width*(-b))
            y2 = int(y0 - width*(a))
            lineEndpoints.append([[x1,y1],[x2,y2]])
            line(videoFrame, (x1,y1), (x2,y2), (0, 0, 255), 2)
            
    for i in range(len(lineEndpoints)):
        for j in range(len(lineEndpoints)):
            if (i==j):
                continue
            xDiff = (lineEndpoints[i][0][0]-lineEndpoints[i][1][0],lineEndpoints[j][0][0]-lineEndpoints[j][1][0])
            yDiff = (lineEndpoints[i][0][1]-lineEndpoints[i][1][1],lineEndpoints[j][0][1]-lineEndpoints[j][1][1])
            div = determinant(xDiff, yDiff)
            if div == 0:
                continue
            d = (determinant(*lineEndpoints[i]), determinant(*lineEndpoints[j]))
            x = int(determinant(d, xDiff) / div)
            y = int(determinant(d, yDiff) / div)
            if x<0 or x>width:
                continue
            if y<0 or y>height:
                continue
            circle(videoFrame, (x,y), radius=0, color=(0, 255, 0), thickness=6)
            
    imshow("Frame", videoFrame)
    if waitKey(1000000) == ord("q"):
        break
    
video.release()
destroyAllWindows()