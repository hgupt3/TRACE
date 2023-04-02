from numpy import pi, ones, zeros, uint8, where, cos, sin
from cv2 import VideoCapture, cvtColor, Canny, line, imshow, waitKey, destroyAllWindows, COLOR_BGR2GRAY, HoughLinesP
from cv2 import threshold, THRESH_BINARY, dilate, floodFill, circle, HoughLines, erode
from TraceHeader import videoFile, checkPath, findIntersection
from CourtMapping import CourtMap

# Retrieve video from video file
video = VideoCapture(videoFile)
checkPath(videoFile)
width = int(video.get(3))
height = int(video.get(4))

# Defining comparison points
NtopLeftP = None
NtopRightP = None
NbottomLeftP = None
NbottomRightP = None

while video.isOpened():
    ret, frame = video.read()
    if frame is None:
        break
    
    # Apply filters that removes noise and simplifies image
    gry = cvtColor(frame, COLOR_BGR2GRAY)
    bw = threshold(gry, 156, 255, THRESH_BINARY)[1]
    canny = Canny(bw, 100, 200)
    
    # Using hough lines probablistic to find lines with most intersections
    hPLines = HoughLinesP(canny, 1, pi/180, threshold=150, minLineLength=100, maxLineGap=10)
    intersectNum = zeros((len(hPLines),2))
    i = 0
    for hPLine1 in hPLines:
        Line1x1, Line1y1, Line1x2, Line1y2 = hPLine1[0]
        Line1 = [[Line1x1,Line1y1],[Line1x2,Line1y2]]
        for hPLine2 in hPLines:
            Line2x1, Line2y1, Line2x2, Line2y2 = hPLine2[0]
            Line2 = [[Line2x1,Line2y1],[Line2x2,Line2y2]]
            if Line1 is Line2:
                continue
            if Line1x1>Line1x2:
                temp = Line1x1
                Line1x1 = Line1x2
                Line1x2 = temp
                
            if Line1y1>Line1y2:
                temp = Line1y1
                Line1y1 = Line1y2
                Line1y2 = temp
                
            intersect = findIntersection(Line1, Line2, Line1x1-200, Line1y1-200, Line1x2+200, Line1y2+200)
            if intersect is not None:
                intersectNum[i][0] += 1
        intersectNum[i][1] = i
        i += 1

    # Lines with most intersections get a fill mask command on them
    i = p = 0
    dilation = dilate(bw, ones((5, 5), uint8), iterations=1)
    nonRectArea = dilation.copy()
    intersectNum = intersectNum[(-intersectNum)[:, 0].argsort()]
    for hPLine in hPLines:
        x1,y1,x2,y2 = hPLine[0]
        # line(frame, (x1,y1), (x2,y2), (255, 255, 0), 2)
        for p in range(8):
            if (i==intersectNum[p][1]) and (intersectNum[i][0]>0):
                # line(frame, (x1,y1), (x2,y2), (0, 0, 255), 2)
                floodFill(nonRectArea, zeros((height+2, width+2), uint8), (x1, y1), 1) 
                floodFill(nonRectArea, zeros((height+2, width+2), uint8), (x2, y2), 1) 
        i+=1
    dilation[where(nonRectArea == 255)] = 0
    dilation[where(nonRectArea == 1)] = 255
    eroded = erode(dilation, ones((5, 5), uint8)) 
    cannyMain = Canny(eroded, 90, 100)
    
    # # Setting variables for extreme lines
    extraLen = width/3
    
    xOLeft = width
    xORight = 0
    xFLeft = width
    xFRight = 0
    xOAxis = [[0,0],[width,0]]
    xFAxis = [[0,height],[width,height]]
    xOLeftLine = [[],[]]
    xORightLine = [[],[]]
    xFLeftLine = [[],[]]
    xFRightLine = [[],[]]

    yOTop = height
    yOBottom = 0
    yFTop = height
    yFBottom = 0
    yOAxis = [[-extraLen,0],[-extraLen,height]]
    yFAxis = [[width+extraLen,0],[width+extraLen,height]]
    yOTopLine = [[],[]]
    yOBottomLine = [[],[]]
    yFTopLine = [[],[]]
    yFBottomLine = [[],[]]
    
    # Finding all lines then allocate them to specified extreme variables
    hLines = HoughLines(cannyMain, 2, pi/180, 250)
    for hLine in hLines:
        for rho,theta in hLine:
            a = cos(theta)
            b = sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + width*(-b))
            y1 = int(y0 + width*(a))
            x2 = int(x0 - width*(-b))
            y2 = int(y0 - width*(a))
            
            # Furthest intersecting point at every axis calculations done here
            intersectxF = findIntersection(xFAxis, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)
            intersectyO = findIntersection(yOAxis, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)
            intersectxO = findIntersection(xOAxis, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)
            intersectyF = findIntersection(yFAxis, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)
            
            if (intersectxO is None) and (intersectxF is None) and (intersectyO is None) and (intersectyF is None):
                continue
            
            if intersectxO is not None:
                if intersectxO[0] < xOLeft:
                    xOLeft = intersectxO[0]
                    xOLeftLine = [[x1,y1],[x2,y2]]
                if intersectxO[0] > xORight:
                    xORight = intersectxO[0]
                    xORightLine = [[x1,y1],[x2,y2]]
            if intersectyO is not None:
                if intersectyO[1] < yOTop:
                    yOTop = intersectyO[1]
                    yOTopLine = [[x1,y1],[x2,y2]]
                if intersectyO[1] > yOBottom:
                    yOBottom = intersectyO[1]
                    yOBottomLine = [[x1,y1],[x2,y2]]
                    
            if intersectxF is not None:
                if intersectxF[0] < xFLeft:
                    xFLeft = intersectxF[0]
                    xFLeftLine = [[x1,y1],[x2,y2]]
                if intersectxF[0] > xFRight:
                    xFRight = intersectxF[0]
                    xFRightLine = [[x1,y1],[x2,y2]]
            if intersectyF is not None:
                if intersectyF[1] < yFTop:
                    yFTop = intersectyF[1]
                    yFTopLine = [[x1,y1],[x2,y2]]
                if intersectyF[1] > yFBottom:
                    yFBottom = intersectyF[1]
                    yFBottomLine = [[x1,y1],[x2,y2]]
            # line(frame, (x1,y1), (x2,y2), (0, 0, 255), 2)
    
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
    #     line(frame, (lineEndpoints[i][0][0],lineEndpoints[i][0][1]), (lineEndpoints[i][1][0],lineEndpoints[i][1][1]), (0, 0, 255), 2)
    
    # Find four corners of the court and display it
    topLeftP = findIntersection(xOLeftLine, yOTopLine, -extraLen, 0, width+extraLen, height)
    topRightP = findIntersection(xORightLine, yFTopLine, -extraLen, 0, width+extraLen, height)
    bottomLeftP = findIntersection(xFLeftLine, yOBottomLine, -extraLen, 0, width+extraLen, height)
    bottomRightP = findIntersection(xFRightLine, yFBottomLine, -extraLen, 0, width+extraLen, height)
        
    # If all corner points are different or something not found, rerun print
    if (not(topLeftP == NtopLeftP)) and (not(topRightP == NtopRightP)) and (not(bottomLeftP == NbottomLeftP)) and (not(bottomRightP == NbottomRightP)):
        line(frame, topLeftP, topRightP, (0, 0, 255), 2)
        line(frame, bottomLeftP, bottomRightP, (0, 0, 255), 2)
        line(frame, topLeftP, bottomLeftP, (0, 0, 255), 2)
        line(frame, topRightP, bottomRightP, (0, 0, 255), 2)
        
        circle(frame, topLeftP, radius=0, color=(255, 0, 255), thickness=10)
        circle(frame, topRightP, radius=0, color=(255, 0, 255), thickness=10)
        circle(frame, bottomLeftP, radius=0, color=(255, 0, 255), thickness=10)
        circle(frame, bottomRightP, radius=0, color=(255, 0, 255), thickness=10)
        
        NtopLeftP = topLeftP
        NtopRightP = topRightP
        NbottomLeftP = bottomLeftP
        NbottomRightP = bottomRightP
        
    else:
        line(frame, NtopLeftP, NtopRightP, (0, 0, 255), 2)
        line(frame, NbottomLeftP, NbottomRightP, (0, 0, 255), 2)
        line(frame, NtopLeftP, NbottomLeftP, (0, 0, 255), 2)
        line(frame, NtopRightP, NbottomRightP, (0, 0, 255), 2)
        
        circle(frame, NtopLeftP, radius=0, color=(255, 0, 255), thickness=10)
        circle(frame, NtopRightP, radius=0, color=(255, 0, 255), thickness=10)
        circle(frame, NbottomLeftP, radius=0, color=(255, 0, 255), thickness=10)
        circle(frame, NbottomRightP, radius=0, color=(255, 0, 255), thickness=10)


    dst = CourtMap(NbottomLeftP, NtopLeftP, NtopRightP, NbottomRightP, frame)
    imshow("Frame", dst)
    if waitKey(1) == ord("q"):
        break
    
video.release()
destroyAllWindows()