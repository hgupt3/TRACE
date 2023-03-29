from skimage.transform import ProjectiveTransform
import numpy as np

def CourtMap(bottom_left, top_left, top_right, bottom_right, data):
    t = ProjectiveTransform()
    src = np.asarray([bottom_left, top_left, top_right, bottom_right])
    dst = np.asarray([[0, 0], [0, 1], [1, 1], [1, 0]])
    if not t.estimate(src, dst): raise Exception("estimate failed")
    data = np.array(data)
    data_transformed = t(data)
    return data_transformed

