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

    not_agreed_lst = [
        (76, [8, 9, 9, 8, 8, 3, 3]),
        (2582, [3, 2, 1, 3, 2, 2, 1]),
        (3490, [6, 5, 2, 6, 2, 2, 6]),
        (4774, [7, 9, 8, 7, 7, 8, 8]),
        (4842, [9, 4, 7, 4, 9, 7, 9]),
        (6070, [3, 2, 3, 2, 8, 3, 2]),
        (6460, [4, 2, 8, 8, 4, 2, 2]),
        (9903, [3, 7, 2, 3, 7, 7, 3]),
        (11821, [3, 6, 5, 5, 6, 6, 3]),
        (11862, [2, 6, 6, 2, 2, 8, 8]),
        (15158, [5, 5, 0, 3, 3, 5, 3]),
        (15712, [5, 0, 8, 5, 8, 2, 5]),
        (16365, [5, 8, 5, 8, 5, 8, 6]),
        (17175, [5, 5, 9, 9, 9, 8, 5]),
        (18669, [9, 9, 4, 4, 8, 9, 8]),
        (19629, [6, 2, 2, 2, 6, 6, 8]),
        (23874, [4, 4, 6, 6, 5, 6, 5]),
        (24180, [3, 6, 3, 6, 0, 0, 3]),
        (24553, [3, 5, 8, 5, 2, 3, 3]),
        (25939, [0, 5, 8, 8, 8, 0, 5]),
        (26213, [0, 9, 7, 7, 0, 7, 0])
    ]
    labels = [str(v) for _, v in not_agreed_lst]
    numbers = [k for k, _ in not_agreed_lst]
    display(X_test[numbers, :], 3, 7, labels=labels)
