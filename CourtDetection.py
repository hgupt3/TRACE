import numpy as np
import matplotlib.pyplot as plt
from cv2 import VideoCapture, cvtColor, Canny, HoughLinesP, line, imshow, waitKey, destroyAllWindows, COLOR_BGR2GRAY
from TraceHeader import videoFile, checkPath

video = VideoCapture(videoFile)
checkPath(videoFile)

while video.isOpened():
    ret, frame = video.read()
    if frame is None:
        break
    frame = cvtColor(frame, COLOR_BGR2GRAY)
    frame = Canny(frame, 200, 500)
    
    ret, videoFrame = video.read()
    if videoFrame is None:
        break
    
    lines = HoughLinesP(frame,rho = 1,theta = np.pi/180,threshold = 300,minLineLength = 100,maxLineGap = 50)

    for i in lines:
        for x1, y1, x2, y2 in i:
            line(videoFrame, (x1, y1), (x2, y2), (20, 20, 220), 3)
    
    imshow("Frame", videoFrame)
    if waitKey(1) == ord("q"):
        break
    
video.release()
frame.release()
videoFrame.release()
destroyAllWindows()