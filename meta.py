#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path


# File names
DATA_DIR = 'data'
TRAIN_CSV_FILENAME = 'train.csv'
TEST_CSV_FILENAME = 'test.csv'
TRAIN_LABELS_BIN_FILENAME = 'train_labels.npy'
TRAIN_PIXELS_BIN_FILENAME = 'train_pixels.npy'
TEST_PIXELS_BIN_FILENAME = 'test_pixels.npy'


def data_filename(fn):
    return path.join(DATA_DIR, fn)


# Images
IMG_WIDTH = 28
IMG_HEIGHT = 28


def img_to_vector(img):
    return img.reshape((IMG_HEIGHT * IMG_WIDTH))


def vector_to_imt(img):
    return img.reshape((IMG_HEIGHT, IMG_WIDTH))
