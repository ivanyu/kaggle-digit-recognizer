#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import meta
from meta import data_filename
from classify_base import enumerate_and_write_predictions
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import Callback, LearningRateScheduler
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras_base import FakeLock, LearningPlotCallback
from keras_base import load_data_prepared_for_keras, make_predictions
from keras_base import save_model


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

# Multilayer perceptron 1 Mk IV - default Keras ImageDataGenerator
# 182 epochs
# Last epoch: 6s - loss: 0.0172 - acc: 0.9994
#                - val_loss: 0.0861 - val_acc: 0.9852
# Train time: ~18,2 minutes
# Test: 0.98557

# Multilayer perceptron 1 Mk V - shorter learning (-30 steps with initial rate)
# ~~ equals to Mk IV

regularization = 0.00001
w_regularizer = l2(regularization)

inputs = Input(shape=(meta.IMG_WIDTH, meta.IMG_HEIGHT, 1))

layer = Flatten()(inputs)

layer = Dense(1000,
              input_shape=(meta.IMG_VEC_LENGTH,),
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

model_json = model.to_json()

nb_epoch = 500
batch_size = 64
nb_classes = 10

X_train, y_train, X_valid, y_valid, X_test =\
    load_data_prepared_for_keras(nb_classes, valid_split=0.1)

samples_per_epoch = X_train.shape[0]


class ValAccuracyEarlyStopping(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (epoch > 150 and logs['val_acc'] >= 0.9850) or epoch > 180:
            self.stopped_epoch = epoch
            self.model.stop_training = True
        pass


class StepsLearningRateScheduler(LearningRateScheduler):
    def __init__(self):
        super(StepsLearningRateScheduler, self).\
            __init__(StepsLearningRateScheduler._schedule)

    @staticmethod
    def _schedule(epoch):
        if epoch < 50:
            return 0.0010
        elif epoch < 80:
            return 0.0009
        elif epoch < 90:
            return 0.0008
        elif epoch < 100:
            return 0.0007
        elif epoch < 110:
            return 0.0006
        elif epoch < 120:
            return 0.0005
        elif epoch < 130:
            return 0.0004
        elif epoch < 140:
            return 0.0003
        elif epoch < 150:
            return 0.0002
        elif epoch < 160:
            return 0.0001
        return 0.00005


learning_plot_callback = LearningPlotCallback(nb_epoch)
val_acc_early_stopping = ValAccuracyEarlyStopping()

learning_rate_scheduler = StepsLearningRateScheduler()

train_start = time.time()

idg = ImageDataGenerator(dim_ordering='tf')
train_iter = idg.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
train_iter.lock = FakeLock()

history = model.fit_generator(
    train_iter,
    samples_per_epoch=samples_per_epoch,
    nb_epoch=nb_epoch,
    verbose=2,
    callbacks=[learning_plot_callback,
               val_acc_early_stopping,
               learning_rate_scheduler],

    validation_data=(X_valid, y_valid)
)

print('Train time, s:', int(time.time() - train_start))

predictions = make_predictions(model, X_test)
print(predictions)
output_file_name = data_filename('play_nn.csv')
print("Writing output file {}...".format(output_file_name))
enumerate_and_write_predictions(predictions, output_file_name)

save_model(model, 'mlp1', 6)
