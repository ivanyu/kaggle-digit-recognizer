#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
import meta
from meta import data_filename


def load_data(scaling=None):
    assert (scaling is None or scaling == 'standard' or
            scaling == 'minmax01' or scaling == 'minmax-11' or
            scaling == 'maxabs')

    print("Loading data...")
    X_train = np.load(data_filename(meta.TRAIN_PIXELS_BIN_FILENAME))
    y_train = np.load(data_filename(meta.TRAIN_LABELS_BIN_FILENAME))
    X_test = np.load(data_filename(meta.TEST_PIXELS_BIN_FILENAME))

    if scaling is not None:
        if scaling == 'standard':
            scaler = StandardScaler(copy=True).fit(X_train)
        elif scaling == 'minmax01':
            scaler = MinMaxScaler(feature_range=(0, 1),
                                  copy=True).fit(X_train)
        elif scaling == 'minmax-11':
            scaler = MinMaxScaler(feature_range=(-1, 1),
                                  copy=True).fit(X_train)
        elif scaling == 'maxabs':
            scaler = MaxAbsScaler().fit(X_train)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

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


def create_pca_applicator(definition):
    """
    :param definition: a string in format "N[w]",
    where N is the number of principal components,
    and "w" is optional whitening flag.
    """

    m = re.match(r'^(?P<n>\d+)(?P<w>w?)$', definition)
    if m is None:
        raise Exception("Incorrect format of PCA definition: {}"
                        .format(definition))

    n = int(m.group('n'))
    whiten = m.group('w') is not None and m.group('w') != ''
    return PCA(n_components=n, whiten=whiten)
