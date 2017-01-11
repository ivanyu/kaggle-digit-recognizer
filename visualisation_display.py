#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
import meta
from meta import data_filename


def display(images, row_n, col_n):
    for i in range(len(images)):
        plt.subplot(row_n, col_n, i + 1)
        pixels = meta.vector_to_imt(images[i, :])
        plt.imshow(pixels, cmap='gray')
    plt.show()


if __name__ == '__main__':
    X_test = np.load(data_filename(meta.TEST_PIXELS_BIN_FILENAME))
    display(X_test[0:100, :], 10, 10)
