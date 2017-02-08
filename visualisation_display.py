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

    not_agreed_lst = [275, 626, 2646, 3097, 3127, 4480, 5571, 5784, 6379,
                      6527, 7063, 10094, 10162, 11139, 11615, 11862,
                      12864, 13905, 15624, 16204, 16452, 17281, 17836,
                      17931, 18669, 19173, 20774, 21546, 22009, 22261, 22555,
                      23874, 24789, 25103, 25523, 25816, 26560, 26650]

    # labels = [str(v) for _, v in not_agreed_lst]
    # numbers = [k for k, _ in not_agreed_lst]
    numbers = not_agreed_lst
    display(X_test[numbers, :], 5, 8,)
