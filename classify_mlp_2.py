#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
from os import path
import time
import numpy as np
from keras.models import Model
from keras.layers import Input, Flatten
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import Callback, LearningRateScheduler
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import meta
from meta import data_filename
from keras_base import LearningPlotCallback
from keras_base import load_data_prepared_for_keras, make_predictions
from keras_base import train_model, train_5_fold_for_stacking, save_model
from classify_base import enumerate_and_write_predictions


# Multilayer perceptron 2
# 282 epochs
# Last epoch: 3s - loss: 0.0284 - acc: 0.9989
#                - val_loss: 0.0832 - val_acc: 0.9876
# Train time: ~15 minutes
# Test: 0.98657

# Multilayer perceptron 2 Mk II - default Keras ImageDataGenerator


def create_model():
    regularization = 0.00001
    w_regularizer = l2(regularization)

    inputs = Input(shape=(meta.IMG_WIDTH, meta.IMG_HEIGHT, 1))

    layer = Flatten()(inputs)

    layer = Dense(800,
                  input_shape=(meta.IMG_WIDTH * meta.IMG_HEIGHT,),
                  init='glorot_normal',
                  activation=None, W_regularizer=w_regularizer)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.4)(layer)

    layer = Dense(800,
                  init='glorot_normal',
                  activation=None, W_regularizer=w_regularizer)(layer)
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
    return model


batch_size = 64


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


def image_data_generator_creator():
    return ImageDataGenerator(dim_ordering='tf')


def train_mk_ii():
    nb_epoch = 500

    X_train, y_train, X_valid, y_valid, X_test =\
        load_data_prepared_for_keras(valid_split=0.1)

    class ValAccuracyEarlyStopping(Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs['val_acc'] >= 0.9875 or epoch >= 300:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            pass

    learning_plot_callback = LearningPlotCallback(nb_epoch)
    val_acc_early_stopping = ValAccuracyEarlyStopping()

    learning_rate_scheduler = StepsLearningRateScheduler()

    train_start = time.time()
    model = train_model(create_model(), nb_epoch, batch_size,
                        X_train, y_train, X_valid, y_valid,
                        image_data_generator_creator,
                        callbacks=[
                            learning_plot_callback,
                            val_acc_early_stopping,
                            learning_rate_scheduler])
    print('Train time, s:', int(time.time() - train_start))

    predictions = make_predictions(model, X_test)
    print(predictions)
    output_file_name = data_filename('play_nn.csv')
    print("Writing output file {}...".format(output_file_name))
    enumerate_and_write_predictions(predictions, output_file_name)

    save_model(model, 'mlp2', 6)


if __name__ == '__main__':
    # train_mk_ii()
    train_5_fold_for_stacking(
        create_model, 'mlp2',
        batch_size,
        nb_epoch=280,
        learning_rate_scheduler=StepsLearningRateScheduler(),
        image_data_generator_creator=image_data_generator_creator,
        model_dir='stacking_models')
