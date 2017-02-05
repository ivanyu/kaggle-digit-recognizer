#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
import meta
from meta import data_filename


def display(images, row_n, col_n, vmin=0.0, vmax=1.0, labels=None):
    for i in range(len(images)):
        plt.subplot(row_n, col_n, i + 1)
        plt.axis('off')
        pixels = meta.vector_to_imt(images[i, :])
        plt.imshow(pixels, cmap='gray', vmin=vmin, vmax=vmax)
        if labels:
            plt.text(0, -2, str(labels[i]))
    plt.show()


if __name__ == '__main__':
    X_test = np.load(data_filename(meta.TEST_PIXELS_BIN_FILENAME))
    display(X_test[0:100, :], 10, 10)
