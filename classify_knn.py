#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import click
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from meta import data_filename
from classify_base import load_data
from classify_base import enumerate_and_write_predictions


@click.group()
def cli():
    pass


option_k = click.option('-k', help='Number of nearest neighbors.',
                        type=int, required=True)


@cli.command()
@click.option('-f', '--folds', help='Number of folds.',
              type=int, required=True)
@option_k
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
@option_k
def classify(k):
    """Classify digits in the test set"""

    (X_train, y_train, X_test) = load_data()

    print("kNN classifier, k =", k)
    classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

    print("Test classification")
    print("Training classifier...")
    time_start = time.time()
    classifier.fit(X_train, y_train)
    print("Finished, took {} s".format(time.time() - time_start))

    length = len(X_test)
    print("Predicting...")
    time_start = time.time()
    predictions = classifier.predict(X_test).reshape((length, 1))
    print("Finished, took {} s".format(time.time() - time_start))
    # print(predictions)

    print("Writing output file...")
    enumerate_and_write_predictions(
        predictions, data_filename('final_knn_{}.csv'.format(k)))


if __name__ == '__main__':
    cli()