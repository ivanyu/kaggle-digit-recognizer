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
from classify_base import create_pca_applicator


@click.group()
def cli():
    pass


option_k = click.option('-k', help='Number of nearest neighbors.',
                        type=int, required=True)
option_pca = click.option('--pca', help='Apply PCA with possible whitening',
                          type=str, required=False, default=None)

@cli.command()
@click.option('-f', '--folds', help='Number of folds.',
              type=int, required=True)
@option_k
@option_pca
def cv(folds, k, pca):
    """Run cross-validation"""

    (X_train, y_train, _) = load_data('minmax01')

    if pca is not None:
        pca_applicator = create_pca_applicator(pca)
        print('Applying PCA {}'.format(pca_applicator))
        pca_applicator.fit(X_train)
        X_train = pca_applicator.transform(X_train)

    print("kNN classifier, k =", k)
    classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    print(classifier)

    print("Cross-validation")
    scores = cross_val_score(classifier, X_train, y_train,
                             cv=folds, n_jobs=3, verbose=50)
    print(scores)


@cli.command()
@option_k
@option_pca
def classify(k, pca):
    """Classify digits in the test set"""

    (X_train, y_train, X_test) = load_data('minmax01')

    if pca is not None:
        pca_applicator = create_pca_applicator(pca)
        print('Applying PCA {}'.format(pca_applicator))
        pca_applicator.fit(X_train)
        X_train = pca_applicator.transform(X_train)
        X_test = pca_applicator.transform(X_test)

    print("kNN classifier, k =", k)
    classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    print(classifier)

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
        predictions, data_filename('final_knn_{}_PCA={}.csv'.format(k, pca)))


if __name__ == '__main__':
    from sys import argv
    cli(argv[1:])
