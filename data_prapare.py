#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import sys
import os
from os import path
import numpy as np
from scipy import ndimage
import scipy.misc
import meta
from meta import data_filename
from visualisation_display import display


def zoom(img, zoom_tuple, **kwargs):
    """
    Zoom image in two dimensions.
    :param img: image
    :param zoom_tuple: zoom factors as (zoom_vertical, zoom_horizontal)
    :return: zoomed image
    """

    assert img.shape == (meta.IMG_WIDTH * meta.IMG_HEIGHT,)
    assert len(zoom_tuple) == 2

    if zoom_tuple[0] == 1 and zoom_tuple[1] == 1:
        return img

    img = img.reshape((meta.IMG_WIDTH, meta.IMG_HEIGHT))

    out = ndimage.zoom(img, zoom_tuple, order=0)

    assert np.all(out >= 0)

    # Crop excess.
    crop_window = [None, None]
    for dim, target_size in [(0, meta.IMG_HEIGHT), (1, meta.IMG_WIDTH)]:
        if out.shape[dim] > target_size:
            to_crop = out.shape[dim] - target_size
            to_crop_1 = int(np.round(to_crop / 2))
            to_crop_2 = to_crop - to_crop_1
            crop_window[dim] = slice(to_crop_1, out.shape[dim] - to_crop_2)
        else:
            crop_window[dim] = slice(0, out.shape[dim])
    out = out[crop_window[0], crop_window[1]]

    # Pad with zeros (black).
    padding_h = max(out.shape[0], meta.IMG_HEIGHT)
    padding_w = max(out.shape[1], meta.IMG_WIDTH)
    padding = np.zeros((padding_h, padding_w))

    to_pad_h = padding_h - out.shape[0]
    to_pad_w = padding_w - out.shape[1]

    to_pad_h_1 = int(np.round(to_pad_h / 2))
    to_pad_h_2 = to_pad_h - to_pad_h_1
    to_pad_w_1 = int(np.round(to_pad_w / 2))
    to_pad_w_2 = to_pad_w - to_pad_w_1

    padding[to_pad_h_1:(padding_h - to_pad_h_2), to_pad_w_1:(padding_w - to_pad_w_2)] = out

    return padding.reshape((meta.IMG_WIDTH * meta.IMG_HEIGHT))

# # Vertical 0.8 - 1.3
# # Horizontal 0.6 - 1.5
# vertical_zoom_range = [i / 10 for i in range(8, 13 + 1)]
# horizontal_zoom_range = [i / 10 for i in range(6, 15 + 1)]

# Vertical 0.9 - 1.1
# Horizontal 0.9 - 1.1
vertical_zoom_range = [0.9, 1.0, 1.1]
horizontal_zoom_range = [0.9, 1.0, 1.1]

multiplier = len(vertical_zoom_range) * len(horizontal_zoom_range)

directory = "zoomed0"

if not path.exists(meta.keras_data_filename(directory)):
    os.mkdir(meta.keras_data_filename(directory))
if not path.exists(meta.keras_data_filename(path.join(directory, 'train'))):
    os.mkdir(meta.keras_data_filename(path.join(directory, 'train')))
if not path.exists(meta.keras_data_filename(path.join(directory, 'train_img'))):
    os.mkdir(meta.keras_data_filename(path.join(directory, 'train_img')))
if not path.exists(meta.keras_data_filename(path.join(directory, 'valid'))):
    os.mkdir(meta.keras_data_filename(path.join(directory, 'valid')))
if not path.exists(meta.keras_data_filename(path.join(directory, 'valid_img'))):
    os.mkdir(meta.keras_data_filename(path.join(directory, 'valid_img')))

X = np.load(data_filename(meta.TRAIN_PIXELS_BIN_FILENAME))
y = np.load(data_filename(meta.TRAIN_LABELS_BIN_FILENAME))

shuffled_indices = np.random.permutation(len(X))
valid_split = 0.1
split_point = int(len(shuffled_indices)*valid_split)
train_indices = shuffled_indices[:-split_point]
valid_indices = shuffled_indices[-split_point:]

for idx in valid_indices:
    img = X[idx, :]
    fname = meta.keras_data_filename(
        '{0}/{1:07d}-{2}.npy'.format(path.join(directory, 'valid'),
                                     idx,
                                     y[idx]))
    if os.path.exists(fname):
        print("File {} exists".format(fname))
    # assert not os.path.exists(fname)
    np.save(fname, img)

    fname_img = meta.keras_data_filename(
        '{0}/{1:07d}-{2}.jpg'.format(path.join(directory, 'valid_img'),
                                     idx,
                                     y[idx]))
    scipy.misc.imsave(fname_img, img.reshape(meta.IMG_WIDTH, meta.IMG_HEIGHT))

for _i, idx in enumerate(train_indices):
    img = X[idx, :]

    if _i % 100 == 0:
        print(_i)

    # if i > 4200:
    #     break

    transformation_counter = 0
    for zoom_vertical in vertical_zoom_range:
        for zoom_horizontal in horizontal_zoom_range:
            img_zoomed = zoom(img, (zoom_vertical, zoom_horizontal))
            fname = meta.keras_data_filename(
                '{0}/{1:05d}-{2:04d}-{3}.npy'.format(path.join(directory, 'train'),
                                                     idx, transformation_counter, y[idx]))
            transformation_counter += 1
            if os.path.exists(fname):
                print("File {} exists".format(fname))
            #assert not os.path.exists(fname)
            np.save(fname, img_zoomed)

            fname_img = meta.keras_data_filename(
                '{0}/{1:05d}-{2:04d}-{3}.jpg'.format(path.join(directory, 'train_img'),
                                                     idx, transformation_counter, y[idx]))
            scipy.misc.imsave(fname_img, img.reshape(meta.IMG_WIDTH, meta.IMG_HEIGHT))
