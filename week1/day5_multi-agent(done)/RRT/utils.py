import math


def on_segment(p, q, r):
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2

def segments_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False

def line_intersects_rectangle(p1, p2, rect):
    x_min, y_min, x_max, y_max = rect
    
    top_left = (x_min, y_max)
    top_right = (x_max, y_max)
    bottom_left = (x_min, y_min)
    bottom_right = (x_max, y_min)
    
    if (segments_intersect(p1, p2, top_left, top_right) or
        segments_intersect(p1, p2, top_right, bottom_right) or
        segments_intersect(p1, p2, bottom_right, bottom_left) or
        segments_intersect(p1, p2, bottom_left, top_left)):
        return True

    return False


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_direction(from_point, to_point):
    vector = (to_point[0] - from_point[0], to_point[1] - from_point[1])
    length = math.sqrt(vector[0]**2 + vector[1]**2)
    return (vector[0] / length, vector[1] / length)
