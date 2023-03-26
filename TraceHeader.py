from os import path
from sys import exit

# Video to be used placed in Clips folder
videoFile = './Videos/Clips/Clip3.mp4'

def checkBounds(frame1, frame2):
    flag = 0
    if (frame1.xcenter == 0):
        if (frame1.x+frame1.xoffset>1):
            print("crop1: x Coordinates out of bounds")
            flag = 1
    else:
        if (frame1.x>1):
            print("crop1: x Coordinates out of bounds")
            flag = 1
    if (frame1.ycenter == 0):
        if (frame1.y+frame1.yoffset>1):
            print("crop1: y Coordinates out of bounds")
            flag = 1
    else:
        if (frame1.y>1):
            print("crop1: y Coordinates out of bounds")
            flag = 1
            
    if (frame2.xcenter == 0):
        if (frame2.x+frame2.xoffset>1):
            print("crop2: x Coordinates out of bounds")
            flag = 1
    else:
        if (frame2.x>1):
            print("crop2: x Coordinates out of bounds")
            flag = 1
    if (frame2.ycenter == 0):
        if (frame2.y+frame2.yoffset>1):
            print("crop2: y Coordinates out of bounds")
            flag = 1
    else:
        if (frame2.y>1):
            print("crop2: y Coordinates out of bounds")
            flag = 1
    if (flag):
        exit()
        
def checkPath(filePath):
    flag = 0
    if not path.exists(filePath):
        print("videoFile: path "+filePath+" does not exist")
        flag = 1
    if (flag):
        exit()
        
def calculatePixels(frame, width, height):
    frame.x = int(width*frame.x)
    frame.y = int(height*frame.y)
    if frame.xcenter:
        frame.xoffset = int((width-frame.x)/2)
    else:
        frame.xoffset = int(width*frame.xoffset)
    if frame.ycenter:
        frame.yoffset = int((height-frame.y)/2)
    else:
        frame.yoffset = int(height*frame.yoffset)
    return frame

def determinant(a, b):
    return a[0] * b[1] - a[1] * b[0]
    
def findIntersection(line1, line2, xStart, yStart, xEnd, yEnd):
    xDiff = (line1[0][0]-line1[1][0],line2[0][0]-line2[1][0])
    yDiff = (line1[0][1]-line1[1][1],line2[0][1]-line2[1][1])
    div = determinant(xDiff, yDiff)
    if div == 0:
        return None
    d = (determinant(*line1), determinant(*line2))
    x = int(determinant(d, xDiff) / div)
    y = int(determinant(d, yDiff) / div)
    if (x<xStart) or (x>xEnd):
        return None
    if (y<yStart) or (y>yEnd):
        return None
    return x,y
