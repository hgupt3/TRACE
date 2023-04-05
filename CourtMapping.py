# from numpy import float32, maximum, minimum, amax
from numpy import max, min, float32
from cv2 import VideoCapture, getPerspectiveTransform, warpPerspective, perspectiveTransform
from TraceHeader import videoFile, checkPath

video = VideoCapture(videoFile)
checkPath(videoFile)
width = int(video.get(3))
height = int(video.get(4))

ratio = (1097/2377)
courtHeight = height * 0.5
courtWidth = courtHeight * ratio
yOffset = ((height - courtHeight) / 2)+130
xOffset = (width - courtWidth) / 2

def courtMap(bottom_left, top_left, top_right, bottom_right, img):
    pts1 = float32([[top_left, top_right, bottom_left, bottom_right]])
    pts2 = float32([[0+xOffset,0+yOffset],[courtWidth+xOffset,0+yOffset],[0+xOffset,courtHeight+yOffset],[courtWidth+xOffset,courtHeight+yOffset]])
    M = getPerspectiveTransform(pts1,pts2)
    dst = warpPerspective(img,M,(width,height))
    
    return dst
