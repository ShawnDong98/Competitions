#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import paddle
import numpy as np
import cv2


def norm_angle(angle, range=[-np.pi / 4, np.pi]):
    return (angle - range[0]) % range[1] + range[0]


# rbox function implemented using numpy
def poly2rbox_le135_np(poly):
    """convert poly to rbox [-pi / 4, 3 * pi / 4]
    Args:
        poly: [x1, y1, x2, y2, x3, y3, x4, y4]
    Returns:
        rbox: [cx, cy, w, h, angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)

    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])

    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) *
                    (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) *
                    (pt2[1] - pt3[1]))

    width = max(edge1, edge2)
    height = min(edge1, edge2)

    rbox_angle = 0
    if edge1 > edge2:
        rbox_angle = np.arctan2(float(pt2[1] - pt1[1]), float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        rbox_angle = np.arctan2(float(pt4[1] - pt1[1]), float(pt4[0] - pt1[0]))

    rbox_angle = norm_angle(rbox_angle)

    x_ctr = float(pt1[0] + pt3[0]) / 2
    y_ctr = float(pt1[1] + pt3[1]) / 2
    return [x_ctr, y_ctr, width, height, rbox_angle]


def poly2rbox_oc_np(poly):
    """convert poly to rbox (0, pi / 2]
    Args:
        poly: [x1, y1, x2, y2, x3, y3, x4, y4]
    Returns:
        rbox: [cx, cy, w, h, angle]
    """
    points = np.array(poly, dtype=np.float32).reshape((-1, 2))
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    # using the new OpenCV Rotated BBox definition since 4.5.1
    # if angle < 0, opencv is older than 4.5.1, angle is in [-90, 0)
    if angle < 0:
        angle += 90
        w, h = h, w

    # convert angle to [0, 90)
    if angle == -0.0:
        angle = 0.0
    if angle == 90.0:
        angle = 0.0
        w, h = h, w

    angle = angle / 180 * np.pi
    return [cx, cy, w, h, angle]


def poly2rbox_np(polys, rbox_type='oc'):
    """
    polys: [x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rboxes: [x_ctr,y_ctr,w,h,angle]
    """
    assert rbox_type in ['oc', 'le135'], 'only oc or le135 is supported now'
    poly2rbox_fn = poly2rbox_oc_np if rbox_type == 'oc' else poly2rbox_le135_np
    rboxes = []
    for poly in polys:
        x, y, w, h, angle = poly2rbox_fn(poly)
        rbox = np.array([x, y, w, h, angle], dtype=np.float32)
        rboxes.append(rbox)

    return np.array(rboxes)


def cal_line_length(point1, point2):
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                 [[x4, y4], [x1, y1], [x2, y2], [x3, y3]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
                 [[x2, y2], [x3, y3], [x4, y4], [x1, y1]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) \
                     + cal_line_length(combinate[i][1], dst_coordinate[1]) \
                     + cal_line_length(combinate[i][2], dst_coordinate[2]) \
                     + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.array(combinate[force_flag]).reshape(8)


def rbox2poly_np(rboxes):
    """
    rboxes:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    polys = []
    for i in range(len(rboxes)):
        x_ctr, y_ctr, width, height, angle = rboxes[i][:5]
        tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
        rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        poly = R.dot(rect)
        x0, x1, x2, x3 = poly[0, :4] + x_ctr
        y0, y1, y2, y3 = poly[1, :4] + y_ctr
        poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
        poly = get_best_begin_point_single(poly)
        polys.append(poly)
    polys = np.array(polys)
    return polys