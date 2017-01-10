#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import meta
from meta import data_filename


csv_train = np.genfromtxt(data_filename(meta.TRAIN_CSV_FILENAME),
                          dtype=int, delimiter=",", skip_header=1)
X_train = csv_train[:, 1:] # pixels
y_train = csv_train[:, 0] # labels

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)

csv_test = np.genfromtxt(data_filename(meta.TEST_CSV_FILENAME),
                         dtype=int, delimiter=",", skip_header=1)
X_test = csv_test
print("X_test: ", X_test.shape)

np.save(data_filename(meta.TRAIN_PIXELS_BIN_FILENAME), X_train)
np.save(data_filename(meta.TRAIN_LABELS_BIN_FILENAME), y_train)

np.save(data_filename(meta.TEST_PIXELS_BIN_FILENAME), X_test)
