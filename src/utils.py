import functools
import itertools
import os
from urllib.parse import urlparse

import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

import tf_utils
import tf_utils.tf_input


def get_domain_from_url(url):
    """Returns the fully-qualified domain fo an url."""
    return urlparse(url).netloc


def zip_dicts(*dicts):
    """Given a list of dictionaries, zip their corresponding
    values and return the resulting dictionary"""
    return {key: [dictionary[key] for dictionary in dicts] for key in dicts[0].keys()}


def dict_combinations(*dict_list_lists):
    """Given a list of lists of dictionaries return the list of
    all possible dictionaries resulting from the unions of sampling
    a dictionary from each list."""
    combinations = itertools.product(*dict_list_lists)  # cartesian product of all
    return map(lambda comb: functools.reduce(lambda a, b: dict(list(a.items()) + list(b.items())),
                                             comb, {}), combinations)  # return resulting dicts


def get_dataset_with_splits(directory):
    """Given a dataset directory, return the dataset as a an X and y. Also return the indices of the
    train,  validation and test samples."""
    # get train, validation and test data
    train_dataset = tf_utils.tf_input.build_dataset(os.path.join(directory, 'dom-full-train-*.csv'), add_weights=False)
    validation_dataset = tf_utils.tf_input.build_dataset(os.path.join(directory, 'dom-full-validation-*.csv'), add_weights=False)
    test_dataset = tf_utils.tf_input.build_dataset(os.path.join(directory, 'dom-full-test-*.csv'), add_weights=False)

    # precache in memory
    train_data = tf_utils.tf_input.data_from_dataset(train_dataset,
                                                     len(dd.read_csv(os.path.join(directory, 'dom-full-train-*.csv'))))
    validation_data = tf_utils.tf_input.data_from_dataset(validation_dataset,
                                                          len(dd.read_csv(os.path.join(directory, 'dom-full-validation-*.csv'))))
    test_data = tf_utils.tf_input.data_from_dataset(test_dataset,
                                                    len(dd.read_csv(os.path.join(directory, 'dom-full-test-*.csv'))))

    # get X and y
    train_X, train_y = train_data[0]['X'], train_data[1]
    validation_X, validation_y = validation_data[0]['X'], validation_data[1]
    test_X, test_y = test_data[0]['X'], test_data[1]

    # get the split indices - then build the slice objects
    split_points = np.cumsum([0, train_y.size, validation_y.size, test_y.size])
    split_slices = [slice(i, j) for i, j in zip(split_points[:-1], split_points[1:])]
    return (np.concatenate((train_X, validation_X, test_X)),
            np.concatenate((train_y, validation_y, test_y)),
            split_slices)


def get_metrics(estimator, big_X, big_y, train_ind, validation_ind, test_ind, hyperparams={}):
    """Returns a datafram of results containing the score for train, test and validation
    for the given estimator. Optionally tunes with a search space of given hyperparameters."""
    # create a big set that is split just by the normal split
    split = [(train_ind, validation_ind)]
    # define the grid search with the goal to maximize f1 score on validation
    grid_search = GridSearchCV(estimator=estimator, param_grid=hyperparams, scoring='f1',
                               cv=split, verbose=2, pre_dispatch=1)
    grid_search.fit(big_X, big_y)

    result_df = pd.DataFrame(data=[
        {'f1-score': f1_score(big_y[train_ind], grid_search.predict(big_X[train_ind, :])), 'set': 'train'},
        {'f1-score': f1_score(big_y[validation_ind], grid_search.predict(big_X[validation_ind, :])),
         'set': 'validation'},
        {'f1-score': f1_score(big_y[test_ind], grid_search.predict(big_X[test_ind, :])), 'set': 'test'}
    ])
    return result_df, grid_search


def dict_combinations(*dict_list_lists):
    """Given a list of lists of ditionaries return the list of
    all posible dictionarie resulting from the unions of sampling
    a dictionary from each list."""
    combinations = itertools.product(*dict_list_lists)  # cartesian product of all
    return map(lambda comb: functools.reduce(lambda a, b: dict(list(a.items()) + list(b.items())),
                                             comb, {}), combinations)  # return resulting dicts