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
from keras_base import LearningPlotCallback
from keras_base import load_data_prepared_for_keras, make_predictions
from keras_base import train_model, train_5_fold_for_stacking, save_model
from classify_base import enumerate_and_write_predictions


def create_model():
    inputs = Input(shape=(meta.IMG_WIDTH, meta.IMG_HEIGHT, 1))

    # regularization = 0.00001
    regularization = 0.0
    w_regularizer = l2(regularization)

    layer = Conv2D(nb_filter=32, nb_row=5, nb_col=5,
                   init='glorot_normal', border_mode='same',
                   input_shape=(meta.IMG_WIDTH, meta.IMG_HEIGHT, 1),
                   activation='elu',
                   dim_ordering='tf')(inputs)
    layer = MaxPooling2D(pool_size=(2, 2), border_mode='same')(layer)

    layer = Conv2D(nb_filter=64, nb_row=5, nb_col=5,
                   init='glorot_normal', border_mode='same',
                   activation='elu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2), border_mode='same')(layer)

    layer = Conv2D(nb_filter=128, nb_row=5, nb_col=5,
                   init='glorot_normal', border_mode='same',
                   activation='elu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2), border_mode='same')(layer)

    layer = Flatten()(layer)

    layer = Dense(800,
                  init='glorot_normal',
                  activation=None, W_regularizer=w_regularizer)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('elu')(layer)
    layer = Dropout(0.1)(layer)

    layer = Dense(800,
                  init='glorot_normal',
                  activation=None, W_regularizer=w_regularizer)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('elu')(layer)
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
    return model


batch_size = 64


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


def image_data_generator_creator():
    return ImageDataGenerator(
        dim_ordering='tf',
        rotation_range=10.0, zoom_range=0.2, shear_range=0.4)


def train():
    nb_epoch = 150

    X_train, y_train, X_valid, y_valid, X_test =\
        load_data_prepared_for_keras(valid_split=0.1)

    class ValAccuracyEarlyStopping(Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs['val_acc'] >= 0.9960 or epoch >= 120:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            pass


    learning_rate_scheduler = StepsLearningRateScheduler()
    val_acc_early_stopping = ValAccuracyEarlyStopping()
    learning_plot_callback = LearningPlotCallback(nb_epoch)

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

    save_model(model, 'cnn1_mk_ii', 0)


if __name__ == '__main__':
    # train()
    # train_5_fold_for_stacking(
    #     create_model, 'cnn2',
    #     batch_size,
    #     nb_epoch=150,
    #     learning_rate_scheduler=StepsLearningRateScheduler(),
    #     image_data_generator_creator=image_data_generator_creator,
    #     model_dir='stacking_models')

    # Pseudo-labeling
    train_5_fold_for_stacking(
        create_model, 'cnn2_psblb',
        batch_size,
        nb_epoch=170,
        learning_rate_scheduler=StepsLearningRateScheduler(),
        image_data_generator_creator=image_data_generator_creator,
        model_dir='stacking_models',
        pseudolabel_fraction=0.25)
