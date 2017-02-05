#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
import meta
from meta import data_filename
from classify_base import enumerate_and_write_predictions
from keras_base import MnistIterators


# Multilayer perceptron 1
# 186 epochs
# Last epoch: 2s - loss: 0.0245 - acc: 0.9989
#                - val_loss: 0.0758 - val_acc: 0.9860
# Train time: ~7 minutes
# Test: 0.98443

# Multilayer perceptron 1 Mk II - less gerularized, iterator and functional API
# 183 epochs
# Last epoch: 4s - loss: 0.0173 - acc: 0.9994
#                - val_loss: 0.0789 - val_acc: 0.9852
# Train time: ~12.2 minutes
# Test: 0.98557

# Multilayer perceptron 1 Mk III - split into train/valid in advance,
# no logic changed
# 185 epochs
# Last epoch: 3s - loss: 0.0168 - acc: 0.9996
#                - val_loss: 0.0786 - val_acc: 0.9850
# Train time: ~10 minutes
# Test: 0.98443


if __name__ == '__main__':
    from keras.models import Model
    from keras.layers import Input, Dense, Activation
    from keras.layers import Dropout, BatchNormalization
    from keras.optimizers import Adam
    from keras.callbacks import Callback, LearningRateScheduler
    from keras.regularizers import l2

    regularization = 0.00001
    w_regularizer = l2(regularization)

    inputs = Input(shape=(meta.IMG_VEC_LENGTH,))

    layer = Dense(1000,
                  input_shape=(meta.IMG_VEC_LENGTH,),
                  init='glorot_normal',
                  activation=None, W_regularizer=w_regularizer)(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.4)(layer)
    outputs = Dense(10,
                    init='glorot_normal',
                    activation='softmax', W_regularizer=w_regularizer)(layer)

    model = Model(input=inputs, output=outputs)

    optimizer = Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.summary()

    n_epochs = 500
    batch_size = 64

    mis = MnistIterators(meta.keras_data_filename('original/'),
                         batch_size=batch_size)

    X_test = np.load(data_filename(meta.TEST_PIXELS_BIN_FILENAME))
    X_test = mis.scaler.transform(X_test)


    class LearningPlotCallback(Callback):
        def __init__(self, n_epochs):
            super(LearningPlotCallback, self).__init__()
            self._n_epochs = n_epochs

        def on_train_begin(self, logs={}):
            plt.ion()
            plt.axis([0, self._n_epochs, 0, 0.5])
            plt.grid()
            self._loss = []
            self._val_loss = []

        def on_epoch_end(self, epoch, logs={}):
            self._loss.append(logs['loss'])
            self._val_loss.append(logs['val_loss'])

            plt.plot(self._loss, 'r-')
            plt.plot(self._val_loss, 'b-')
            plt.pause(0.0001)
            pass


    class ValAccuracyEarlyStopping(Callback):
        def on_epoch_end(self, epoch, logs={}):
            if epoch > 180 and logs['val_acc'] >= 0.9850:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            pass


    class StepsLearningRateScheduler(LearningRateScheduler):
        def __init__(self):
            super(StepsLearningRateScheduler, self).\
                __init__(StepsLearningRateScheduler._schedule)

        @staticmethod
        def _schedule(epoch):
            if epoch < 80:
                return 0.0010
            elif epoch < 110:
                return 0.0009
            elif epoch < 120:
                return 0.0008
            elif epoch < 130:
                return 0.0007
            elif epoch < 140:
                return 0.0006
            elif epoch < 150:
                return 0.0005
            elif epoch < 160:
                return 0.0004
            elif epoch < 170:
                return 0.0003
            elif epoch < 180:
                return 0.0002
            elif epoch < 190:
                return 0.0001
            return 0.00005


    learning_plot_callback = LearningPlotCallback(n_epochs)
    val_acc_early_stopping = ValAccuracyEarlyStopping()

    learning_rate_scheduler = StepsLearningRateScheduler()

    train_start = time.time()

    history = model.fit_generator(
        mis.create_train_iterator(),
        mis.samples_per_epoch,
        nb_epoch=n_epochs,
        verbose=2,
        callbacks=[learning_plot_callback,
                   val_acc_early_stopping,
                   learning_rate_scheduler],

        validation_data=mis.get_validation_data()
    )

    print('Train time, s:', int(time.time() - train_start))

    proba = model.predict_on_batch(X_test)
    if proba.shape[-1] > 1:
        predictions = proba.argmax(axis=-1)
    else:
        predictions = (proba > 0.5).astype('int32')

    predictions = predictions.reshape((predictions.shape[0], 1))
    print(predictions)
    output_file_name = data_filename(
        'play_nn.csv'.format()
    )
    print("Writing output file {}...".format(output_file_name))
    enumerate_and_write_predictions(predictions, output_file_name)
