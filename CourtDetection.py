from numpy import pi, ones, zeros, uint8, where
import matplotlib.pyplot as plt
from cv2 import VideoCapture, cvtColor, Canny, line, imshow, waitKey, destroyAllWindows, COLOR_BGR2GRAY, HoughLinesP
from cv2 import threshold, THRESH_BINARY, dilate, floodFill, circle, HoughLines
from TraceHeader import videoFile, checkPath, findIntersection

# Retrieve video from video file
video = VideoCapture(videoFile)
checkPath(videoFile)
width = int(video.get(3))
height = int(video.get(4))
    
while video.isOpened():
    ret, frame = video.read()
    if frame is None:
        break
    
    # Apply filters that removes noise and simplifies image
    gry = cvtColor(frame, COLOR_BGR2GRAY)
    bw = threshold(gry, 156, 255, THRESH_BINARY)[1]
    canny = Canny(bw, 100, 200)
    
    # Using hough lines probablistic to find lines with most intersections
    hPLines = HoughLinesP(canny, 1, pi/180, threshold=200, minLineLength=150, maxLineGap=20)
    intersectNum = zeros((len(hPLines),2))
    i = 0
    for hPLine1 in hPLines:
        for Line1x1,Line1y1,Line1x2,Line1y2 in hPLine1:
            for hPLine2 in hPLines:
                for Line2x1,Line2y1,Line2x2,Line2y2 in hPLine2:
                    if (Line1x1 == Line2x1) and (Line1y1 == Line2y1)and (Line1x2 == Line2x2) and (Line1y2 == Line2y2):
                        continue
                    if Line1x1>Line1x2:
                        temp = Line1x1
                        Line1x1 = Line1x2
                        Line1x2 = temp
                        
                    if Line1y1>Line1y2:
                        temp = Line1y1
                        Line1y1 = Line1y2
                        Line1y2 = temp
                        
                    intersect = findIntersection(([Line1x1,Line1y1],[Line1x2,Line1y2]), ([Line2x1,Line2y1],[Line2x2,Line2y2]), Line1x1-200, Line1y1-200, Line1x2+200, Line1y2+200)
                    intersectNum[i][1] = i
                    if intersect is not None:
                        intersectNum[i][0] += 1
        i += 1

    i = p = 0
    dilation = dilate(bw, ones((5, 5), uint8), iterations=1)
    nonRectArea = dilation.copy()
    
    intersectNum = intersectNum[(-intersectNum)[:, 0].argsort()]
    for hPLine in hPLines:
        for x1,y1,x2,y2 in hPLine:
            line(frame, (x1,y1), (x2,y2), (255, 255, 0), 2)
            for p in range(8):
                if (i==intersectNum[p][1]) and (intersectNum[i][0]>1):
                    line(frame, (x1,y1), (x2,y2), (0, 0, 255), 2)
                    floodFill(nonRectArea, zeros((height+2, width+2), uint8), (x1, y1), 1) 
                    floodFill(nonRectArea, zeros((height+2, width+2), uint8), (x2, y2), 1) 
        i+=1
    
    dilation[where(nonRectArea == 255)] = 0
    dilation[where(nonRectArea == 1)] = 255
    canny = Canny(dilation, 100, 200)
    
    # hLines = HoughLines(canny, 2, np.pi/180, 350)
    
    # xOLeft = width
    # xORight = 0
    # xFLeft = width
    # xFRight = 0
    # xOAxis = [[0,0],[width,0]]
    # xFAxis = [[0,height],[width,height]]
    # xOLeftLine = [[],[]]
    # xORightLine = [[],[]]
    # xFLeftLine = [[],[]]
    # xFRightLine = [[],[]]

    # yOTop = height
    # yOBottom = 0
    # yFTop = height
    # yFBottom = 0
    # yOAxis = [[0,0],[0,height]]
    # yFAxis = [[width,0],[width,height]]
    # yOTopLine = [[],[]]
    # yOBottomLine = [[],[]]
    # yFTopLine = [[],[]]
    # yFBottomLine = [[],[]]
    
    # for hLine in hLines:
    #     for rho,theta in hLine:
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a*rho
    #         y0 = b*rho
    #         x1 = int(x0 + width*(-b))
    #         y1 = int(y0 + width*(a))
    #         x2 = int(x0 - width*(-b))
    #         y2 = int(y0 - width*(a))
            
    #         intersectxO = findIntersection(xOAxis, [[x1,y1],[x2,y2]], 0, 0, width, height)
    #         intersectxF = findIntersection(xFAxis, [[x1,y1],[x2,y2]], 0, 0, width, height)
    #         intersectyO = findIntersection(yOAxis, [[x1,y1],[x2,y2]], 0, 0, width, height)
    #         intersectyF = findIntersection(yFAxis, [[x1,y1],[x2,y2]], 0, 0, width, height)
            
    #         if (intersectxO is None) and (intersectxF is None) and (intersectyO is None) and (intersectyF is None):
    #             continue
            
    #         if intersectxO is not None:
    #             if intersectxO[0] < xOLeft:
    #                 xOLeft = intersectxO[0]
    #                 xOLeftLine = [[x1,y1],[x2,y2]]
    #             if intersectxO[0] > xORight:
    #                 xORight = intersectxO[0]
    #                 xORightLine = [[x1,y1],[x2,y2]]
    #         if intersectyO is not None:
    #             if intersectyO[1] < yOTop:
    #                 yOTop = intersectyO[1]
    #                 yOTopLine = [[x1,y1],[x2,y2]]
    #             if intersectyO[1] > yOBottom:
    #                 yOBottom = intersectyO[1]
    #                 yOBottomLine = [[x1,y1],[x2,y2]]
                    
    #         if intersectxF is not None:
    #             if intersectxF[0] < xFLeft:
    #                 xFLeft = intersectxF[0]
    #                 xFLeftLine = [[x1,y1],[x2,y2]]
    #             if intersectxF[0] > xFRight:
    #                 xFRight = intersectxF[0]
    #                 xFRightLine = [[x1,y1],[x2,y2]]
    #         if intersectyF is not None:
    #             if intersectyF[1] < yFTop:
    #                 yFTop = intersectyF[1]
    #                 yFTopLine = [[x1,y1],[x2,y2]]
    #             if intersectyF[1] > yFBottom:
    #                 yFBottom = intersectyF[1]
    #                 yFBottomLine = [[x1,y1],[x2,y2]]
    #         line(videoFrame, (x1,y1), (x2,y2), (0, 0, 255), 2)
    
    # lineEndpoints = []
    
    # lineEndpoints.append(xOLeftLine)
    # lineEndpoints.append(xORightLine)
    # lineEndpoints.append(yOTopLine)
    # lineEndpoints.append(yOBottomLine)
    # lineEndpoints.append(xFLeftLine)
    # lineEndpoints.append(xFRightLine)
    # lineEndpoints.append(yFTopLine)
    # lineEndpoints.append(yFBottomLine)
    
    # for i in range(len(lineEndpoints)):
    #     line(videoFrame, (lineEndpoints[i][0][0],lineEndpoints[i][0][1]), (lineEndpoints[i][1][0],lineEndpoints[i][1][1]), (0, 0, 255), 2)
    
    # topLeftP = findIntersection(xOLeftLine, yOTopLine, 0, 0, width, height)
    # topRightP = findIntersection(xORightLine, yFTopLine, 0, 0, width, height)
    # bottomLeftP = findIntersection(xFLeftLine, yOBottomLine, 0, 0, width, height)
    # bottomRightP = findIntersection(xFRightLine, yFBottomLine, 0, 0, width, height)
    
    # line(videoFrame, topLeftP, topRightP, (0, 0, 255), 2)
    # line(videoFrame, bottomLeftP, bottomRightP, (0, 0, 255), 2)
    # line(videoFrame, topLeftP, bottomLeftP, (0, 0, 255), 2)
    # line(videoFrame, topRightP, bottomRightP, (0, 0, 255), 2)
    
    # circle(videoFrame, topLeftP, radius=0, color=(255, 0, 255), thickness=10)
    # circle(videoFrame, topRightP, radius=0, color=(255, 0, 255), thickness=10)
    # circle(videoFrame, bottomLeftP, radius=0, color=(255, 0, 255), thickness=10)
    # circle(videoFrame, bottomRightP, radius=0, color=(255, 0, 255), thickness=10)
            
    imshow("Frame", dilation)
    if waitKey(1) == ord("q"):
        break
    
video.release()
destroyAllWindows()