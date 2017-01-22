#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import click
from click.parser import OptionParser
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from meta import data_filename
from classify_base import load_data
from classify_base import enumerate_and_write_predictions
from classify_base import create_pca_applicator


CONTEXT_SETTINGS = dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
)


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


def create_classifier(svm_type, c, **kwargs):
    assert svm_type == 'linear' or svm_type == 'rbf'

    if svm_type == 'linear':
        return LinearSVC(C=c,
                         penalty="l2",
                         loss="squared_hinge",
                         dual=False, # better False if n_samples > n_features
                         multi_class='ovr',
                         verbose=0)
    elif svm_type == 'rbf':
        return SVC(C=c,
                   kernel='rbf',
                   gamma=kwargs['gamma'],
                   cache_size=2000)


def parse_rbf_options(ctx):
    option_parser = OptionParser()
    option_parser.add_option(['-g', '--gamma'], 'gamma')
    additional_opts, _, _ = option_parser.parse_args(ctx.args)
    if 'gamma' in additional_opts:
        additional_opts['gamma'] = float(additional_opts['gamma'])
    else:
        additional_opts['gamma'] = None
    return additional_opts


option_svm_type = click.argument('svm_type',
                                 type=click.Choice(['linear', 'rbf']),
                                 required=True)
option_c = click.option('-c',
                        help='Penalty parameter C of the error term.',
                        type=float, required=True)
option_gamma = click.option('-g', '--gamma',
                            help='',
                            type=float, required=False, default=None)
option_dual = click.option('-dual',
                           help='Select the algorithm to either solve '
                                'the dual or primal optimization problem. '
                                'Prefer dual=False when '
                                'n_samples > n_features.',
                           type=bool, default=False, required=False)
option_pca = click.option('--pca', help='Apply PCA with possible whitening',
                          type=str, required=False, default=None)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.pass_context
@option_svm_type
@click.option('-f', '--folds', help='Number of folds.',
              type=int, required=True)
@option_c
@option_pca
def cv(ctx, svm_type, folds, c, pca):
    """Run cross-validation"""

    (X_train, y_train, _) = load_data('minmax01')

    additional_opts = {}
    additional_line = ""
    if svm_type == "rbf":
        additional_opts = parse_rbf_options(ctx)
        gamma = additional_opts['gamma']
        additional_line = ", gamma = {}".format(gamma)

    if pca is not None:
        pca_applicator = create_pca_applicator(pca)
        print('Applying PCA {}'.format(pca_applicator))
        pca_applicator.fit(X_train)
        X_train = pca_applicator.transform(X_train)

    print("SVM classifier {}, C = {}{}".format(
        svm_type, c, additional_line))
    classifier = create_classifier(svm_type, c, **additional_opts)
    print(classifier)

    print("Cross-validation")
    scores = cross_val_score(classifier, X_train, y_train,
                             cv=folds, n_jobs=3, verbose=50)
    print(scores)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.pass_context
@option_svm_type
@option_c
@option_pca
def classify(ctx, svm_type, c, pca):
    """Classify digits in the test set"""

    (X_train, y_train, X_test) = load_data('minmax01')

    additional_opts = {}
    additional_line_log = ""
    additional_line_filename = ""
    if svm_type == "rbf":
        additional_opts = parse_rbf_options(ctx)
        gamma = additional_opts['gamma']
        additional_line_log = ", gamma = {}".format(gamma)
        additional_line_filename = "_gamma={}".format(gamma)

    print("SVM classifier {}, C = {}{}".format(
        svm_type, c, additional_line_log))
    classifier = create_classifier(svm_type, c, **additional_opts)

    if pca is not None:
        pca_applicator = create_pca_applicator(pca)
        print('Applying PCA {}'.format(pca_applicator))
        pca_applicator.fit(X_train)
        X_train = pca_applicator.transform(X_train)
        X_test = pca_applicator.transform(X_test)

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

    output_file_name = data_filename(
        'final_svm_{}_C={}{}_pca={}.csv'.format(
            svm_type, c, additional_line_filename, pca)
    )
    print("Writing output file {}...".format(output_file_name))
    enumerate_and_write_predictions(predictions, output_file_name)


@cli.command(context_settings=CONTEXT_SETTINGS)
def gridsearch():
    print("Started: ", time.localtime())

    # c_range = [2**(-5), 2**(-3), 2**-1, 1, 2**3, 2**5,
    #            2**7, 2**9, 2**11, 2**13, 2**15]
    c_range = [100]
    # gamma_range = [0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060]
    # gamma_range = [0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049]
    # gamma_range = [0.0470, 0.0471, 0.0472, 0.0473, 0.0474, 0.0475,
    #                0.0476, 0.0477, 0.0478, 0.0479,
    #                0.0480, 0.0481, 0.0482, 0.0483, 0.0484, 0.0485,
    #                0.0486, 0.0487, 0.0488, 0.0489, 0.0490]
    # gamma_range = [0.04740, 0.04743, 0.0475, 0.04747, 0.04749, 0.04750,
    #                0.04753, 0.04755, 0.04757, 0.04759, 0.04760, 0.04763,
    #                0.04765, 0.04767, 0.04769, 0.04770, 0.04773, 0.04775,
    #                0.04777, 0.04779, 0.04780]
    gamma_range = [0.047400, 0.047401, 0.047402, 0.047403, 0.047404, 0.047405,
                   0.047406, 0.047407, 0.047408, 0.047409, 0.047410,
                   0.047411, 0.047412, 0.047413, 0.047414, 0.047415, 0.047416,
                   0.047417, 0.047418, 0.047419, 0.047420]


    # c_range = [5, 10, 20, 30, 40]
    # c_range = [2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5]
    # gamma_range = [0.0474]
    c_range = [64]
    gamma_range = [0.03125] #0.980047619048



    c_range = [1000]
    # gamma_range = [0.1, 1.0, 10.0]
    # gamma_range = [0.45, 0.46, 0.47, 0.48, 0.49, 0.050, 0.051, 0.052, 0.053, 0.054, 0.055]
    # gamma_range = [0.0500, 0.0501, 0.0502, 0.0503, 0.0504, 0.0505,
    #                0.0506, 0.0507, 0.0508, 0.0509, 0.0510,
    #                0.0511, 0.0512, 0.0513, 0.0514, 0.0515,
    #                0.0516, 0.0517, 0.0518, 0.0519, 0.0520]
    # gamma_range = [0.0499, 0.0500, 0.0501]
    gamma_range = [0.35, 0.40, 0.45, 0.049]
    param_grid = [
        # {'kernel': ['linear'],
        #  'C': c_range},

        {'kernel': ['rbf'],
         'C': c_range,
         'gamma': gamma_range},

        # {'kernel': ['poly'],
        #  'C': c_range,
        #  'degree': [1, 2, 3, 4, 5]},
    ]
    #param_grid = [{'kernel': ['linear'], 'C': [1.0]}]
    svc = SVC()
    grid_search = GridSearchCV(estimator=svc,
                               param_grid=param_grid,
                               n_jobs=-1,
                               refit=False)

    (X_train, y_train, X_test) = load_data('minmax01')
    # (X_train, y_train, X_test) = load_data('standard')
    import meta
    from sklearn.decomposition import PCA
    pca = PCA(n_components=35, whiten=True)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    grid_search.fit(X_train, y_train)

    print(grid_search)
    print("cv_results_: ", grid_search.cv_results_)
    print("best_score_: ", grid_search.best_score_)
    #print("best_estimator_: ", grid_search.best_estimator_)
    print("best_params_: ", grid_search.best_params_)
    print("best_index_: ", grid_search.best_index_)
    print("scorer_: ", grid_search.scorer_)
    print("n_splits_: ", grid_search.n_splits_)

    print("Finished: ", time.localtime())

if __name__ == '__main__':
    from sys import argv
    cli(argv[1:])
