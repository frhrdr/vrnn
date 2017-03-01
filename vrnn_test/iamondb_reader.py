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


def mat_to_plot(mat, meanx=0, meany=0, stdx=1, stdy=1):

    # renorm
    mat[:, 0] = (mat[:, 0] + meanx) * stdx
    mat[:, 1] = (mat[:, 1] + meany) * stdy

    for idx in range(2, mat.shape[0]):
        mat[idx, :2] = mat[idx, :2] + mat[idx - 1, :2]

    mat[:, 1] = - mat[:, 1]  # flip y axis for accurate plot


    stroke_ends = np.argwhere(mat[:, 2])  # single out individual strokes
    begin = 0
    print(stroke_ends)
    for end in stroke_ends:
        print(end)
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
    for f in matches:
        mat = xml_to_mat(f)
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
    # plt.hist(idx_mat, 50, normed=1, facecolor='green', alpha=0.75)
    # plt.show()
    for idx in range(1, idx_mat.shape[0]):
        idx_mat[idx] = idx_mat[idx] + idx_mat[idx - 1]

    return seq_mat, idx_mat


def load_and_cut_sequences(source_dir, seq_file='sequences.npy', idx_file='sequence_indices.npy', cut_len=500,
                           normalize=True, mask=True, mask_value=500):

    if not mask:
        mask_value = 0

    seq_mat, idx_mat = load_sequences(source_dir, seq_file, idx_file)

    if normalize:
        mat = seq_mat.astype(float)
        mean = np.mean(mat, axis=0)
        std = np.std(mat, axis=0)
        for idx in [0, 1]:
            mat[:, idx] = (mat[:, idx] - mean[idx]) / std[idx]

        print('normalized data check:')
        print(np.mean(mat, axis=0))
        print(np.std(mat, axis=0))
    else:
        mean = [0, 0]
        std = [1, 1]
    split_list = np.split(seq_mat, idx_mat[1:], axis=0)

    for idx, mat in enumerate(split_list):
        if mat.shape[0] < cut_len:
            padded = np.zeros((cut_len, 3), dtype=float) + mask_value
            padded[:mat.shape[0], :] = mat
            split_list[idx] = padded
        else:
            split_list[idx] = mat[:cut_len, :]

    data_mat = np.asarray(split_list)
    data_mat = np.swapaxes(data_mat, 0, 1)
    return data_mat, mean, std


def normalize_data(data_mat):
    mat = data_mat.astype(float)
    m = np.mean(mat, axis=(0, 1))
    s = np.std(mat, axis=(0, 1))
    print(mat.shape)
    print('mean: ', m.shape, m)
    print('sigm: ', s.shape, s)
    for idx in [0, 1]:
        mat[:, :, idx] = (mat[:, :, idx] - m[idx]) / s[idx]
    return mat, m, s


def no_values_check(val):
    seq, idx = load_sequences('data/handwriting')
    seq = seq[:, :2]
    seq = (seq == val)
    seq = np.sum(seq, 1)
    print(np.sum(seq))
    print(np.sum(seq == 2*val))

# no_values_check(500)

# a01-000u-01
# a = xml_to_mat('data/handwriting/strokesu.xml')
# mat_to_plot(a)
# parse_data_set('data/handwriting')
# print(load_and_cut_sequences('data/handwriting').shape)
mat, mean, std = load_and_cut_sequences('data/handwriting', cut_len=200, mask=False, normalize=True)
# a, m, s = normalize_data(a)
#
np.save('data/handwriting/rough_cut_200_pad_0_max_300_norm_xyonly.npy', mat[:, :, :2])


# mean 200cut: [ 7.60117317  0.3098164]
# std  200cut: [ 33.65332993  34.89729551]