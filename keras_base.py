#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import re
import random
import json
import numpy as np
from keras.preprocessing.image import Iterator
from sklearn.preprocessing import MinMaxScaler
import meta


class _SubsetIterator(Iterator):
    def __init__(self, directory, filenames, classes, nb_class, batch_size,
                 shuffle, scaler):
        self._directory = directory
        self._filenames = np.array(filenames)
        self._classes = np.array(classes)
        self._nb_class = nb_class
        self._scaler = scaler
        samples_n = len(filenames)
        self._cache = {}

        super(_SubsetIterator, self).__init__(
            samples_n, batch_size, shuffle, seed=None)

    def next(self):
        # with self.lock:
        index_array, current_index, current_batch_size =\
            next(self.index_generator)

        batch_x = np.zeros((current_batch_size, meta.IMG_HEIGHT * meta.IMG_WIDTH))
        for i, j in enumerate(index_array):
            fname = self._filenames[j]
            x = self._load_image(fname)
            x = self._scaler.transform(x)
            batch_x[i] = x

        batch_y = np.zeros((len(batch_x), self._nb_class), dtype='float32')
        for i, label in enumerate(self._classes[index_array]):
            batch_y[i, label] = 1.

        return batch_x, batch_y

    def _load_image(self, fname):
        if fname not in self._cache:
            with open(os.path.join(self._directory, fname), 'rb') as fh:
                x = np.frombuffer(fh.read(), dtype='int32').reshape((meta.IMG_WIDTH * meta.IMG_HEIGHT,))
            x = np.expand_dims(x, axis=0)
            self._cache[fname] = x
        return self._cache[fname]


class MnistIterators(object):

    def __init__(self, directory, batch_size, validation_split):
        self._directory = directory
        self._batch_size = batch_size

        scaler_filename = '_scaler.json'

        all_files = os.listdir(directory)
        if scaler_filename in all_files:
            all_files.remove(scaler_filename)

        random.shuffle(all_files)
        split = int(len(all_files) * validation_split)
        self._train_files = all_files[0:-split]
        self._valid_files = all_files[-split:]

        self.scaler = MinMaxScaler(feature_range=(0, 1), copy=False)

        r = re.compile(r'^\d+-(?P<c>\d)')

        class_set = set()

        scaler_filepath = os.path.join(directory, scaler_filename)
        scaler_exists = os.path.exists(scaler_filepath)

        self._train_classes = []
        self._valid_classes = []
        for fname in self._train_files:
            m = r.match(fname)
            assert m.group('c') is not None
            cls = int(m.group('c'))
            class_set.add(cls)
            self._train_classes.append(cls)

            if not scaler_exists:
                x = self._load_image(fname)
                self.scaler.partial_fit(x)

        for fname in self._valid_files:
            m = r.match(fname)
            assert m.group('c') is not None
            cls = int(m.group('c'))
            class_set.add(cls)
            self._valid_classes.append(cls)

            if not scaler_exists:
                x = self._load_image(fname)
                self.scaler.partial_fit(x)

        self._nb_class = len(class_set)
        assert self._nb_class == 10

        self.samples_per_epoch = len(self._train_files)
        self.nb_val_samples = len(self._valid_files)

        if not scaler_exists:
            scaler_params = {
                'scale_': list(self.scaler.scale_),
                'min_': list(self.scaler.min_),
                'n_samples_seen_': self.scaler.n_samples_seen_,
                'data_min_': list(self.scaler.data_min_),
                'data_max_': list(self.scaler.data_max_),
                'data_range_': list(self.scaler.data_range_),
            }
            with open(scaler_filepath, 'w') as f:
                json.dump(scaler_params, f, indent=4)
        else:
            with open(scaler_filepath, 'r') as f:
                scaler_params = json.load(f)
                self.scaler.scale_ = np.array(scaler_params['scale_'])
                self.scaler.min_ = np.array(scaler_params['min_'])
                self.scaler.n_samples_seen_ = scaler_params['n_samples_seen_']
                self.scaler.data_min_ = np.array(scaler_params['data_min_'])
                self.scaler.data_max_ = np.array(scaler_params['data_max_'])
                self.scaler.data_range_ = np.array(scaler_params['data_range_'])

    def create_train_iterator(self):
        return _SubsetIterator(self._directory, self._train_files,
                               self._train_classes, self._nb_class,
                               self._batch_size, shuffle=True,
                               scaler=self.scaler)

    def create_validation_iterator(self):
        result = self.get_validation_data()
        while True:
            yield result

    def get_validation_data(self):
        xs = []
        ys = []
        for fname, label in zip(self._valid_files, self._valid_classes):
            x = self._load_image(fname)
            x = self.scaler.transform(x)
            xs.append(x)
            y = np.zeros(self._nb_class, dtype='float32')
            y[label] = 1.
            ys.append(y)
            # y = np.expand_dims(y, axis=0)
        return (np.vstack(xs), np.vstack(ys))

    def _load_image(self, fname):
        with open(os.path.join(self._directory, fname), 'rb') as fh:
            x = np.frombuffer(fh.read(), dtype='int32').reshape(
                (meta.IMG_WIDTH * meta.IMG_HEIGHT,))
        return np.expand_dims(x, axis=0)
