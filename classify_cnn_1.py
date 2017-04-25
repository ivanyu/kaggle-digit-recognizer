#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
from keras.models import Model
from keras.layers import Input, Flatten
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import Callback, LearningRateScheduler
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import meta
from meta import data_filename
from keras_base import LearningPlotCallback, FakeLock
from keras_base import load_data_prepared_for_keras, make_predictions
from keras_base import save_model
from classify_base import enumerate_and_write_predictions


# CNN 1 Mk I - no regularisation or dropout
# 101 epochs
# Last epoch: 8s - loss: 0.0034 - acc: 0.9987
#                - val_loss: 0.0358 - val_acc: 0.9940
# Train time: ~14 minutes
# Test: 0.99429

# CNN 1 Mk II - no regularisation, dropout 0.1


inputs = Input(shape=(meta.IMG_WIDTH, meta.IMG_HEIGHT, 1))

# regularization = 0.00001
regularization = 0.0
w_regularizer = l2(regularization)

layer = Conv2D(nb_filter=32, nb_row=3, nb_col=3,
               init='glorot_normal', border_mode='same',
               input_shape=(meta.IMG_WIDTH, meta.IMG_HEIGHT, 1),
               activation='relu',
               dim_ordering='tf')(inputs)
layer = MaxPooling2D(pool_size=(2, 2), border_mode='same')(layer)

layer = Conv2D(nb_filter=64, nb_row=3, nb_col=3,
               init='glorot_normal', border_mode='same',
               activation='relu')(layer)
layer = MaxPooling2D(pool_size=(2, 2), border_mode='same')(layer)

layer = Flatten()(layer)

layer = Dense(800,
              init='glorot_normal',
              activation=None, W_regularizer=w_regularizer)(layer)
layer = BatchNormalization()(layer)
layer = Activation('relu')(layer)
layer = Dropout(0.1)(layer)

layer = Dense(800,
              init='glorot_normal',
              activation=None, W_regularizer=w_regularizer)(layer)
layer = BatchNormalization()(layer)
layer = Activation('relu')(layer)
layer = Dropout(0.1)(layer)

outputs = Dense(10,
                init='glorot_normal',
                activation='softmax', W_regularizer=w_regularizer)(layer)

model = Model(input=inputs, output=outputs)

optimizer = Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()

nb_epoch = 150
batch_size = 64

nb_classes = 10

X_train, y_train, X_valid, y_valid, X_test =\
    load_data_prepared_for_keras(nb_classes, valid_split=0.1)

samples_per_epoch = X_train.shape[0]


class ValAccuracyEarlyStopping(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs['val_acc'] >= 0.9960 or epoch >= 120:
            self.stopped_epoch = epoch
            self.model.stop_training = True
        pass


class StepsLearningRateScheduler(LearningRateScheduler):
    def __init__(self):
        super(StepsLearningRateScheduler, self).\
            __init__(StepsLearningRateScheduler._schedule)

    @staticmethod
    def _schedule(epoch):
        if epoch < 30:
            return 0.0010
        elif epoch < 40:
            return 0.0009
        elif epoch < 50:
            return 0.0008
        elif epoch < 60:
            return 0.0007
        elif epoch < 70:
            return 0.0006
        elif epoch < 80:
            return 0.0005
        elif epoch < 90:
            return 0.0004
        elif epoch < 100:
            return 0.0003
        elif epoch < 110:
            return 0.0002
        elif epoch < 120:
            return 0.0001
        return 0.00005

learning_rate_scheduler = StepsLearningRateScheduler()
val_acc_early_stopping = ValAccuracyEarlyStopping()
learning_plot_callback = LearningPlotCallback(nb_epoch)

idg = ImageDataGenerator(dim_ordering='tf',
                         rotation_range=10.0, zoom_range=0.2, shear_range=0.4)
train_iter = idg.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
train_iter.lock = FakeLock()

train_start = time.time()

history = model.fit_generator(
    train_iter,
    samples_per_epoch=samples_per_epoch,
    nb_epoch=nb_epoch,
    verbose=2,
    callbacks=[learning_plot_callback,
               val_acc_early_stopping,
               learning_rate_scheduler
               ],

    validation_data=(X_valid, y_valid)
)

print('Train time, s:', int(time.time() - train_start))

predictions = make_predictions(model, X_test)
print(predictions)
output_file_name = data_filename('play_nn.csv')
print("Writing output file {}...".format(output_file_name))
enumerate_and_write_predictions(predictions, output_file_name)

save_model(model, 'cnn1', 6)
