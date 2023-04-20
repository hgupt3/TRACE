import math

def euclideanDistance(point1, point2):
    return math.dist(point1, point2)

def withinCircle(center, radius, point):
    return radius > euclideanDistance(center, point)

def closestPoint(center, prevPoint, currPoint):
    if euclideanDistance(center, prevPoint) <= euclideanDistance(center, currPoint):
        return True