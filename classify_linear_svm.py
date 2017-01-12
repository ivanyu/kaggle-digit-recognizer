#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import click
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from meta import data_filename
from classify_base import load_data
from classify_base import enumerate_and_write_predictions


@click.group()
def cli():
    pass


def create_classifier(c, dual):
    # dual - better False if n_samples > n_features

    return LinearSVC(C=c,
                     penalty="l2",
                     loss="squared_hinge",
                     dual=dual,
                     multi_class='ovr',
                     verbose=0)


option_c = click.option('-c',
                        help='Penalty parameter C of the error term.',
                        type=float, required=True)
option_dual = click.option('-dual',
                           help='Select the algorithm to either solve '
                                'the dual or primal optimization problem. '
                                'Prefer dual=False when '
                                'n_samples > n_features.',
                           type=bool, default=False, required=False)


@cli.command()
@click.option('-f', '--folds', help='Number of folds.',
              type=int, required=True)
@option_c
@option_dual
def cv(folds, c, dual):
    """Run cross-validation"""

    (X_train, y_train, X_test) = load_data()

    print("SVM classifier, C = {}, dual = {}".format(c, dual))
    classifier = create_classifier(c, dual)

    print("Cross-validation")
    scores = cross_val_score(classifier, X_train, y_train,
                             cv=folds, n_jobs=3, verbose=50)
    print(scores)


@cli.command()
@option_c
@option_dual
def classify(c, dual):
    """Classify digits in the test set"""

    (X_train, y_train, X_test) = load_data()

    print("SVM classifier, C = {}, dual = {}".format(c, dual))
    classifier = create_classifier(c, dual)

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
        predictions, data_filename(
            'final_linearsvm_C={}_dual={}.csv'.format(c, dual)
        )
    )


if __name__ == '__main__':
    cli()
