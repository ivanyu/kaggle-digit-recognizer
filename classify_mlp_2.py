#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import Callback, LearningRateScheduler
from keras.regularizers import l2
from keras.utils import np_utils
import matplotlib.pyplot as plt
import meta
from meta import data_filename
from classify_base import load_data
from classify_base import enumerate_and_write_predictions


# Multilayer perceptron 2
# 282 epochs
# Last epoch: 3s - loss: 0.0284 - acc: 0.9989
#                - val_loss: 0.0832 - val_acc: 0.9876
# Train time: ~15 minutes
# Test: 0.98657


regularization = 0.00001
w_regularizer = l2(regularization)

model = Sequential()
model.add(Dense(800,
                input_shape=(meta.IMG_WIDTH * meta.IMG_HEIGHT,),
                init='glorot_normal',
                activation=None, W_regularizer=w_regularizer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(800,
                init='glorot_normal',
                activation=None, W_regularizer=w_regularizer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(10,
                init='glorot_normal',
                activation='softmax', W_regularizer=w_regularizer))

optimizer = Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()

n_epochs = 500
batch_size = 64

nb_classes = 10
(X_train, y_train, X_test) = load_data('minmax01')
y_train = np_utils.to_categorical(y_train, nb_classes)


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


class ValAccuracyEarlyStopping(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs['val_acc'] >= 0.9875:
            self.stopped_epoch = epoch
            self.model.stop_training = True
        pass


class StepsLearningRateScheduler(LearningRateScheduler):
    def __init__(self):
        super(StepsLearningRateScheduler, self).\
            __init__(StepsLearningRateScheduler._schedule)

    @staticmethod
    def _schedule(epoch):
        if epoch < 120:
            return 0.0010
        elif epoch < 150:
            return 0.0009
        elif epoch < 170:
            return 0.0008
        elif epoch < 190:
            return 0.0007
        elif epoch < 210:
            return 0.0006
        elif epoch < 230:
            return 0.0005
        elif epoch < 250:
            return 0.0004
        elif epoch < 270:
            return 0.0003
        elif epoch < 290:
            return 0.0002
        elif epoch < 310:
            return 0.0001
        return 0.00005


learning_plot_callback = LearningPlotCallback(n_epochs)
val_acc_early_stopping = ValAccuracyEarlyStopping()

learning_rate_scheduler = StepsLearningRateScheduler()

train_start = time.time()

history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    nb_epoch=n_epochs, batch_size=batch_size,
                    callbacks=[learning_plot_callback,
                               val_acc_early_stopping,
                               learning_rate_scheduler
                               ],
                    verbose=2, shuffle=True)

print('Train time, s:', int(time.time() - train_start))

predictions = model.predict_classes(X_test)
predictions = predictions.reshape((predictions.shape[0], 1))
print(predictions)
output_file_name = data_filename(
    'play_nn.csv'.format()
)
print("Writing output file {}...".format(output_file_name))
enumerate_and_write_predictions(predictions, output_file_name)
