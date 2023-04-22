# from numpy import float32, maximum, minimum, amax
from numpy import float32, zeros,  uint8, shape, array, squeeze
from cv2 import VideoCapture, getPerspectiveTransform, warpPerspective, line, circle, rectangle, perspectiveTransform
from TraceHeader import videoFile, checkPath

video = VideoCapture(videoFile)
checkPath(videoFile)
frameWidth = int(video.get(3))
frameHeight = int(video.get(4))

widthP = int(967/1.5)
heightP = int(1585/1.5)

width = int(967/1.5)
height = int(1585/1.5)

ratio = (1097/2377)
courtHeight = int(height * 0.6)
courtWidth = int(courtHeight * ratio)
yOffset = int((height - courtHeight) / 2)
xOffset = int((width - courtWidth) / 2)

courtTL = [xOffset,yOffset]
courtTR = [courtWidth+xOffset,yOffset]
courtBL = [xOffset,courtHeight+yOffset]
courtBR = [courtWidth+xOffset,courtHeight+yOffset]

def courtMap(frame, top_left, top_right, bottom_left, bottom_right):
    pts1 = float32([[top_left, top_right, bottom_left, bottom_right]])
    pts2 = float32([courtTL, courtTR, courtBL, courtBR])
    M = getPerspectiveTransform(pts1,pts2)
    dst = warpPerspective(frame,M,(width,height))
    return dst, M

def showLines(frame):
    rectangle(frame, (0,0),(width,height),(255,0,0),6)
    line(frame, courtTL, courtTR, (0, 0, 255), 2)
    line(frame, courtBL, courtBR, (0, 0, 255), 2)
    line(frame, courtTL, courtBL, (0, 0, 255), 2)
    line(frame, courtTR, courtBR, (0, 0, 255), 2)
    return frame

def showPoint(frame, M, point):
    points = float32([[point]])
    transformed = perspectiveTransform(points, M)[0][0]
    circle(frame, (int(transformed[0]), int(transformed[1])), radius=0, color=(0, 0, 255), thickness=20)
    return frame

def givePoint(M, point):
    points = float32([[point]])
    transformed = perspectiveTransform(points, M)[0][0]
    return (int(transformed[0]), int(transformed[1]))
