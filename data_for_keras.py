#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
import scipy
import scipy.misc
import meta
from meta import keras_data_filename
from classify_base import load_data


if not os.path.exists(meta.KERAS_DATA_DIR):
    os.mkdir(meta.KERAS_DATA_DIR)

if not os.path.exists(keras_data_filename('original')):
    os.mkdir(keras_data_filename('original'))
# if not os.path.exists(keras_data_filename('bin')):
#     os.mkdir(keras_data_filename('bin'))
# if not os.path.exists(keras_data_filename('generator')):
#     os.mkdir(keras_data_filename('generator'))
# for i in range(9 + 1):
#     if not os.path.exists(keras_data_filename('generator/{}'.format(i))):
#         os.mkdir(keras_data_filename('generator/{}'.format(i)))

(X_train, y_train, _) = load_data(None)

for i in range(X_train.shape[0]):
    fname = 'original/{0:07d}-{1}.npy'.format(i, y_train[i])
    x = X_train[i,:]
    np.save(keras_data_filename(fname), x)
# for i in range(X_train.shape[0]):
#     fname = 'bin/{0:07d}-{1}.bin'.format(i, y_train[i])
#     x = X_train[i,:]
#     with open(keras_data_filename(fname), 'wb') as fh:
#         # fh.write(b'{0:s} {1:d} {2:d}\n'.format(x.dtype, *x.shape))
#         fh.write(x.data)
#         fh.flush()
# for i in range(X_train.shape[0]):
#     fname = 'generator/{0}/{1:07d}.jpg'.format(y_train[i], i)
#     x = X_train[i,:].reshape((meta.IMG_HEIGHT, meta.IMG_WIDTH))
#     scipy.misc.imsave(keras_data_filename(fname), x)
