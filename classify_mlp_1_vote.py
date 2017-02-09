#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from os import path
from collections import Counter
import meta
import numpy as np
from keras_base import load_data_prepared_for_keras, make_predictions
from classify_base import enumerate_and_write_predictions
from ensemble_base import load_nn_model


_, _, _, _, X_test = load_data_prepared_for_keras(nb_classes=10,
                                                  valid_split=0.1)


model_numbers = [0, 1, 2, 3, 4, 5, 6]

models = []
predictions = []
for n in model_numbers:
    fname_model = path.join(meta.MODELS_DIR,
                            'model_mlp1_{}.json'.format(n))
    fname_model_weights = path.join(meta.MODELS_DIR,
                                    'model_mlp1_{}_weights.h5'.format(n))
    model = load_nn_model(fname_model, fname_model_weights)
    models.append(model)

    ps = make_predictions(model, X_test)
    ps = ps.reshape((ps.shape[0],))
    predictions.append(ps)

# agree = 0
not_agree = 0
not_agreed_idx = []
final_predictions = []
for idx in range(len(predictions[0])):
    c = Counter()
    for i in model_numbers:
        p = predictions[i][idx]
        c[p] += 1

    final_predictions.append(c.most_common(1)[0][0])

    if c.most_common(1)[0][1] < 4:
        not_agree += 1
        not_agreed_idx.append(idx)
    pass

print(not_agree)
# print(not_agreed_idx)
print(final_predictions)

final_predictions = np.array(final_predictions).reshape(X_test.shape[0], 1)

output_file_name = meta.data_filename('play_nn_vote.csv')
enumerate_and_write_predictions(final_predictions, output_file_name)
