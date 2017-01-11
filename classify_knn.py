#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import click
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import meta
from meta import data_filename


def load_data():
    print("Loading data...")
    X_train = np.load(data_filename(meta.TRAIN_PIXELS_BIN_FILENAME))
    y_train = np.load(data_filename(meta.TRAIN_LABELS_BIN_FILENAME))
    X_test = np.load(data_filename(meta.TEST_PIXELS_BIN_FILENAME))
    print("Data loaded")
    return X_train, y_train, X_test


@click.group()
def cli():
    pass


@cli.command()
@click.option('-f', '--folds', help='Number of folds.', type=int)
@click.option('-k', help='Number of nearest neighbors.', type=int)
def cv(folds, k):
    """Run cross-validation"""

    (X_train, y_train, X_test) = load_data()

    print("kNN classifier, k =", k)
    classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

    print("Cross-validation")
    scores = cross_val_score(classifier, X_train, y_train,
                             cv=folds, n_jobs=3, verbose=50)
    print(scores)


@cli.command()
@click.option('-k', help='Number of nearest neighbors.', type=int)
def classify(k):
    """Classify digits in the test set"""

    (X_train, y_train, X_test) = load_data()

    print("kNN classifier, k =", k)
    classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

    print("Test classification")
    print("Training classifier...")
    classifier.fit(X_train, y_train)
    length = len(X_test)
    print("Predicting...")
    predictions = classifier.predict(X_test).reshape((length, 1))
    # print(predictions)
    print("Writing output file...")
    test_numbers = np.arange(1, length + 1, dtype=int).reshape((length, 1))
    # print(test_numbers)
    predictions_with_numbers = np.hstack((test_numbers, predictions))
    with open(data_filename('final_knn_{}.csv'.format(k)), 'wb') as f:
        f.write(b'ImageId,Label\n')
        np.savetxt(f, predictions_with_numbers, fmt="%i", delimiter=',')
    print()


if __name__ == '__main__':
    cli()
