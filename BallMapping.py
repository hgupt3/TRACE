import math

def euclideanDistance(point1, point2):
    return math.dist(point1, point2)

def withinCircle(center, radius, point):
    return radius > euclideanDistance(center, point)

def closestPoint(prevCenter, currCenter, prevPoint, currPoint):
    if euclideanDistance(prevCenter, prevPoint) <= euclideanDistance(currCenter, currPoint):
        return True