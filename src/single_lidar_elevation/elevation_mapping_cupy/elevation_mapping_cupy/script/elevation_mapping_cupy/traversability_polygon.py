#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import numpy as np
import cupy as cp


def convex_hull(points):
    points = np.unique(points, axis=0)
    if len(points) < 3:
        return None

    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = np.asarray(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return None
    return hull


def get_masked_traversability(map_array, mask, traversability):
    traversability = traversability[1:-1, 1:-1]
    is_valid = map_array[2][1:-1, 1:-1]
    mask = mask[1:-1, 1:-1]

    # invalid place is 0 traversability value
    untraversability = cp.where(is_valid > 0.5, 1 - traversability, 0)
    masked = untraversability * mask
    masked_isvalid = is_valid * mask
    return masked, masked_isvalid


def is_traversable(masked_untraversability, thresh, min_thresh, max_over_n):
    untraversable_thresh = 1 - thresh
    max_thresh = 1 - min_thresh
    over_thresh = cp.where(masked_untraversability > untraversable_thresh, 1, 0)
    polygon = calculate_untraversable_polygon(over_thresh)
    max_untraversability = masked_untraversability.max()
    if over_thresh.sum() > max_over_n:
        is_safe = False
    elif max_untraversability > max_thresh:
        is_safe = False
    else:
        is_safe = True
    return is_safe, polygon


def calculate_area(polygon):
    area = 0
    for i in range(len(polygon)):
        p1 = polygon[i - 1]
        p2 = polygon[i]
        area += (p1[0] * p2[1] - p1[1] * p2[0]) / 2.0
    return abs(area)


def calculate_untraversable_polygon(over_thresh):
    x, y = cp.where(over_thresh > 0.5)
    points = cp.stack([x, y]).T
    hull = convex_hull(cp.asnumpy(points))
    if hull is None:
        return None
    return cp.asarray(np.vstack([hull, hull[0]]))


def transform_to_map_position(polygon, center, cell_n, resolution):
    polygon = center.reshape(1, 2) + (polygon - cell_n / 2.0) * resolution
    return polygon


def transform_to_map_index(points, center, cell_n, resolution):
    indices = ((points - center.reshape(1, 2)) / resolution + cell_n / 2).astype(cp.int32)
    return indices


if __name__ == "__main__":
    polygon = [[0, 0], [2, 0], [0, 2]]
    print(calculate_area(polygon))

    under_thresh = cp.zeros((20, 20))
    # under_thresh[10:12, 8:10] = 1.0
    under_thresh[14:18, 8:10] = 1.0
    under_thresh[1:8, 2:9] = 1.0
    print(under_thresh)
    polygon = calculate_untraversable_polygon(under_thresh)
    print(polygon)
    transform_to_map_position(polygon, cp.array([0.5, 1.0]), 6.0, 0.05)
