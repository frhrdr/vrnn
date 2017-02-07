from __future__ import print_function
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

def xml_to_mat(xml_path, interpolate=False):

    if interpolate:
        raise NotImplementedError

    root = ET.parse(xml_path).getroot()

    stroke_set = root.find('StrokeSet')
    stroke_mat_list = []
    for stroke in stroke_set:
        point_list = []
        for point in stroke:
            coords = (point.attrib['x'], point.attrib['y'], 0)
            point_list.append(coords)
        stroke_mat = np.asarray(point_list, dtype=int)
        stroke_mat[-1, 2] = 1  # mark end of character

        stroke_mat_list.append(stroke_mat)

    mat = np.concatenate(stroke_mat_list, axis=0)

    mat[1:, :2] = mat[1:, :2] - mat[:-1, :2]
    mat[0, :2] = 0

    return mat


def mat_to_img(mat):
    mat = mat[:, :2]
    for idx in range(2, mat.shape[0]):
        mat[idx, :] = mat[idx, :] + mat[idx - 1, :]

    xy_min = np.min(mat, axis=0)

    mat[:, 0] = mat[:, 0] - xy_min[0]
    mat[:, 1] = mat[:, 1] - xy_min[1]

    mat = mat / 10

    xy_max = np.max(mat, axis=0)
    image = np.ones(xy_max+1, dtype=int)

    for idx in range(mat.shape[0]):
        image[mat[idx, 0], mat[idx, 1]] = 0
        pass

    im = plt.imshow(image.T, cmap='gray')
    plt.show(im)

# a01-000u-01
a = xml_to_mat('data/handwriting/strokesu.xml')
mat_to_img(a)


# to save:
# assert stats.find('SensorLocation').attrib['corner'] == 'top_left'  # for now. can easily be addressed
#     sensor_location = stats.find('SensorLocation').attrib['corner']
#     sensor_location = sensor_location.split('_')
#
#     x_min = int(stats.find('VerticallyOppositeCoords').attrib['x'])
#     x_max = int(stats.find('DiagonallyOppositeCoords').attrib['x'])
#     y_min = int(stats.find('HorizontallyOppositeCoords').attrib['y'])
#     y_max = int(stats.find('DiagonallyOppositeCoords').attrib['x'])
#
#     stroke_set = root.find('StrokeSet')
#     stroke_mat_list = []
#     for stroke in stroke_set:
#         point_list = []
#         for point in stroke:
#             coords = (point.attrib['x'], point.attrib['y'], 0)
#             point_list.append(coords)
#         stroke_mat = np.asarray(point_list, dtype=int)
#         stroke_mat[:, 0] = stroke_mat[:, 0] - x_min
#         stroke_mat[:, 1] = stroke_mat[:, 1] - y_min
#         stroke_mat[-1, 2] = 1  # mark end of character
#
#         z = np.zeros(stroke_mat.shape[0],)
#         x_check_mat = (stroke_mat[:, 0] >= z) * (stroke_mat[:, 0] <= (z + x_max - x_min))
#         y_check_mat = (stroke_mat[:, 1] >= z) * (stroke_mat[:, 1] <= (z + y_max - y_min))
#         check_mat = x_check_mat * y_check_mat
#         if not np.all(check_mat):
#             raise ValueError('Values out of bounds')
#
#         stroke_mat_list.append(stroke_mat)
#
#     mat = np.concatenate(stroke_mat_list, axis=0)
