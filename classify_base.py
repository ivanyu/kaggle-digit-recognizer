#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import meta
from meta import data_filename


def load_data():
    print("Loading data...")
    X_train = np.load(data_filename(meta.TRAIN_PIXELS_BIN_FILENAME))
    y_train = np.load(data_filename(meta.TRAIN_LABELS_BIN_FILENAME))
    X_test = np.load(data_filename(meta.TEST_PIXELS_BIN_FILENAME))
    print("Data loaded")
    return X_train, y_train, X_test


def enumerate_and_write_predictions(predictions, fname):
    length = len(predictions)
    test_numbers = np.arange(1, length + 1, dtype=int).reshape((length, 1))
    # print(test_numbers)
    predictions_with_numbers = np.hstack((test_numbers, predictions))
    with open(fname, 'wb') as f:
        f.write(b'ImageId,Label\n')
        np.savetxt(f, predictions_with_numbers, fmt="%i", delimiter=',')
