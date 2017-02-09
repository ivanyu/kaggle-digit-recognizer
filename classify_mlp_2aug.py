#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
from keras.models import Model
from keras.layers import Input, Flatten
from keras.layers import Dense, Activation, Dropout, BatchNormalization
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


# Multilayer perceptron 2
# 282 epochs
# Last epoch: 3s - loss: 0.0284 - acc: 0.9989
#                - val_loss: 0.0832 - val_acc: 0.9876
# Train time: ~15 minutes
# Test: 0.98657

# Multilayer perceptron 2 Mk II - default Keras ImageDataGenerator

# Multilayer perceptron 2 Mk III - data augmentation, no dropout
# 135 epochs
# Last epoch: 6s - loss: 0.0591 - acc: 0.9915
#                - val_loss: 0.0678 - val_acc: 0.9917
# Train time: ~13,5 minutes
# Test: 0.99000

# Multilayer perceptron 2 Mk IV - more data augmentation, no dropout
# 301 epochs
# Last epoch: 6s - loss: 0.0310 - acc: 0.9967
#                - val_loss: 0.0535 - val_acc: 0.9929
# Train time: ~30 minutes
# Test: 0.99171


regularization = 0.000005
w_regularizer = l2(regularization)

inputs = Input(shape=(meta.IMG_WIDTH, meta.IMG_HEIGHT, 1))

layer = Flatten()(inputs)

layer = Dense(800,
              input_shape=(meta.IMG_WIDTH * meta.IMG_HEIGHT,),
              init='glorot_normal',
              activation=None, W_regularizer=w_regularizer)(layer)
layer = BatchNormalization()(layer)
layer = Activation('relu')(layer)
# layer = Dropout(0.4)(layer)

layer = Dense(800,
              init='glorot_normal',
              activation=None, W_regularizer=w_regularizer)(layer)
layer = BatchNormalization()(layer)
layer = Activation('relu')(layer)
# layer = Dropout(0.4)(layer)

outputs = Dense(10,
                init='glorot_normal',
                activation='softmax', W_regularizer=w_regularizer)(layer)

model = Model(input=inputs, output=outputs)

optimizer = Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()

nb_epoch = 500
batch_size = 64

nb_classes = 10

X_train, y_train, X_valid, y_valid, X_test =\
    load_data_prepared_for_keras(nb_classes, valid_split=0.1)

samples_per_epoch = X_train.shape[0]


class ValAccuracyEarlyStopping(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs['val_acc'] >= 0.9930 or epoch >= 350:
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


learning_plot_callback = LearningPlotCallback(nb_epoch)
val_acc_early_stopping = ValAccuracyEarlyStopping()

learning_rate_scheduler = StepsLearningRateScheduler()

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

save_model(model, 'mlp2aug', 6)
