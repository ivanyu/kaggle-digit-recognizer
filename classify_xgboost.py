#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
from os import path
import time
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from classify_base import load_data, enumerate_and_write_predictions
import meta
import numpy as np
import xgboost as xgb


MODEL_DIR = 'stacking_models'


# XGBoost
# 300 rounds
# Train time: ~19.5 minutes
# Test: 0.97371


def train_stacking(clf_name, num_boost_round, params):
    params = params.copy()
    params['objective'] = 'multi:softprob'

    (X_train_original, y_train_original, X_test_original) = load_data('minmax01')
    X_train_all = X_train_original
    X_test = X_test_original

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kfold.split(np.arange(0, X_train_original.shape[0])))

    for (a, b) in folds:
        assert np.all(np.sort(np.concatenate((a, b))) ==
                      np.arange(0, X_train_original.shape[0]))

    fold_pred_file_name = path.join(MODEL_DIR, '{}_folds.npy'.format(clf_name))
    if not path.exists(fold_pred_file_name):
        stacking_train = np.zeros((X_train_original.shape[0], meta.N_CLASSES))

        for fold_n, (train_idxs, val_idxs) in enumerate(folds):
            X_train = X_train_all[train_idxs]
            y_train = y_train_original[train_idxs]
            X_val = X_train_all[val_idxs]
            y_val = y_train_original[val_idxs]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val)

            train_start = time.time()
            clf = xgb.train(params, dtrain, num_boost_round,
                            callbacks=[lambda env: print(fold_n, env)])
            print('Train time, s:', int(time.time() - train_start))

            probabilities = clf.predict(dval)
            stacking_train[val_idxs, :] = probabilities

            if probabilities.shape[-1] > 1:
                predictions = probabilities.argmax(axis=-1)
            else:
                predictions = (probabilities > 0.5).astype('int32')
            accuracy = accuracy_score(y_val, predictions)
            print(accuracy)

        np.save(fold_pred_file_name, stacking_train)

    full_pred_file_name = path.join(MODEL_DIR, '{}_full.npy'.format(clf_name))
    if not path.exists(full_pred_file_name):
        print("Full")

        dtrain_all = xgb.DMatrix(X_train_all, label=y_train_original)
        dtest = xgb.DMatrix(X_test)

        train_start = time.time()

        clf = xgb.train(params, dtrain_all, num_boost_round,
                        callbacks=[lambda env: print(env)])
        print('Train time, s:', int(time.time() - train_start))

        stacking_test = clf.predict(dtest)
        np.save(full_pred_file_name, stacking_test)


def train_simple(clf_name, num_boost_round, params):
    params = params.copy()
    params['objective'] = 'multi:softmax'

    (X_train, y_train, X_test) = load_data('minmax01')
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    time_start = time.time()
    clf = xgb.train(params, dtrain, num_boost_round,
                    callbacks=[lambda env: print(env)])
    print("Finished, took {} s".format(time.time() - time_start))

    predictions = clf.predict(dtest).reshape((X_test.shape[0], 1))
    print("Writing output file...")
    enumerate_and_write_predictions(predictions, 'play_xgboost.csv')

if __name__ == '__main__':
    if not path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    clf_name = "xgb1"
    num_boost_round = 300
    params = {
        #'objective': 'multi:softprob',
        'num_class': meta.N_CLASSES,
        'updater': 'grow_gpu'
    }

    train_stacking(clf_name, num_boost_round, params)
    # train_simple(clf_name, num_boost_round, params)
