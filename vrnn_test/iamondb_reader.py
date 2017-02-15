from __future__ import print_function
import os
import fnmatch
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

def xml_to_mat(xml_path, interpolate=False, max_dist=300):

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

    mat = np.maximum(mat, - max_dist)
    mat = np.minimum(mat, max_dist)
    return mat


def mat_to_plot(mat):

    for idx in range(2, mat.shape[0]):
        mat[idx, :2] = mat[idx, :2] + mat[idx - 1, :2]

    mat[:, 1] = - mat[:, 1]  # flip y axis for accurate plot

    stroke_ends = np.argwhere(mat[:, 2])  # single out individual strokes
    begin = 0
    for end in stroke_ends:
        end = int(end)
        plt.axis('equal')
        plt.plot(mat[begin:end, 0], mat[begin:end, 1], c='blue')
        begin = end + 1
    plt.show()


def parse_data_set(target_dir, root_dir='data/handwriting/xml_data_root/lineStrokes'):
    # for file in os.listdir(root_dir):
    #     if file.endswith('.xml'):
    #         print(file)
    #         break
    matches = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*.xml'):
            matches.append(os.path.join(root, filename))
    print(len(matches))
    print(matches[0])

    mat_list = []
    for file in matches:
        mat = xml_to_mat(file)
        mat_list.append(mat)
        if len(mat_list) % 100 == 0:
            print('loaded ' + str(len(mat_list)) + '/' + str(len(matches)) + ' files')

    len_list = [k.shape[0] for k in mat_list]
    sequence_indices = np.asarray(len_list)
    sequences = np.concatenate(mat_list, axis=0)
    np.save(target_dir + '/sequence_indices.npy', sequence_indices)
    np.save(target_dir + '/sequences.npy', sequences)

    plt.hist(len_list, 50, normed=1, facecolor='green', alpha=0.75)
    plt.show()


def load_sequences(source_dir, seq_file='sequences.npy', idx_file='sequence_indices.npy'):
    seq_mat = np.load(source_dir + '/' + seq_file)
    idx_mat = np.load(source_dir + '/' + idx_file)
    plt.hist(idx_mat, 50, normed=1, facecolor='green', alpha=0.75)
    plt.show()
    for idx in range(1, idx_mat.shape[0]):
        idx_mat[idx] = idx_mat[idx] + idx_mat[idx - 1]

# a01-000u-01
# a = xml_to_mat('data/handwriting/strokesu.xml')
# mat_to_plot(a)
# parse_data_set('data/handwriting')
load_sequences('data/handwriting')


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
