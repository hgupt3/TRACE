import math

def euclideanDistance(point1, point2):
    return math.dist(point1, point2)

# Flag is used to override whether the point is within the circle or not
def withinCircle(center, radius, point, flag):
    if flag:
        return False
    return radius > euclideanDistance(center, point)

def closestPoint(center, prevPoint, currPoint):
    if euclideanDistance(center, prevPoint) < euclideanDistance(center, currPoint):
        return True