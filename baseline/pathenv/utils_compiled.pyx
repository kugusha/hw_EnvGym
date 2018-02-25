import collections
from scipy.spatial.distance import euclidean
import numpy as np
cimport numpy as np


DIFFS = np.array([
        [-1, 0],
        [0,  1],
        [1,  0],
        [0, -1],
], dtype = 'int32')

DTYPE = np.int
ctypedef np.int_t DTYPE_t


def build_distance_map(np.ndarray[np.float_t, ndim = 2] local_map, np.ndarray[DTYPE_t] finish):
    cdef np.ndarray[np.float_t, ndim = 2] result = -np.array(local_map)

    queue = collections.deque()
    queue.append(((finish[0], finish[1]), 0))
    result[finish[0], finish[1]] = 0

    cdef DTYPE_t new_y, new_x, max_y = local_map.shape[0], max_x = local_map.shape[1], off_i, cur_x, cur_y
    cdef np.float_t new_dist, cur_dist

    cdef DTYPE_t all_dx[4]
    cdef DTYPE_t all_dy[4]
    cdef DTYPE_t all_diff[4]
    all_dx[:] = [0, 1, 0, -1]
    all_dy[:] = [-1, 0, 1, 0]

    while len(queue) > 0:
        (cur_y, cur_x), cur_dist = queue.popleft()
        off_i = 0
        for off_i in range(4):
            new_y = cur_y + all_dy[off_i]
            new_x = cur_x + all_dx[off_i]

            # we are not going to obstacle
            if (0 <= new_y < max_y and 0 <= new_x < max_x
                and (new_y != finish[0] or new_x != finish[1])):

                new_dist = cur_dist + 1

                if result[new_y, new_x] == 0 or result[new_y, new_x] > new_dist:
                    queue.append(((new_y, new_x), new_dist))
                    result[new_y, new_x] = new_dist
    return result


def check_finish_achievable(np.ndarray[np.float_t, ndim = 2] local_map, np.ndarray[DTYPE_t] start, np.ndarray[DTYPE_t] finish):
    if  np.allclose(start, finish):
        return True
    return build_distance_map(local_map, finish)[start[0], start[1]] > 0


def line_intersection(tuple line1, tuple line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(tuple a, tuple b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    y = det(d, ydiff) / div
    x = det(d, xdiff) / div

    return x, y