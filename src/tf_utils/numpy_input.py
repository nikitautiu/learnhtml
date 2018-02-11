import os

import numpy as np
from dask import dataframe as dd


def get_numpy_dataset(csv_pattern):
    """Given a csv pattern for a dataset, return a dict of ('id', 'X', 'y') containing the components of
    the dataset.
    :param csv_pattern: the pattern of the csv files
    :return: the specified dict
    """
    ddf = dd.read_csv(csv_pattern)

    # separate the numpy arrays
    id_arr = ddf[['url', 'path']].values.compute()
    X_arr = ddf.drop(['url', 'path', 'content_label'], axis=1).values.compute()
    y_arr = ddf['content_label'].values.compute()

    return {'id': id_arr, 'X': X_arr, 'y': y_arr}


def get_numpy_datasets(directory):
    """
    Given a dataset directory, return the dataset as a an X and y. Also return the indices of the
    train,  validation and test samples. Keeps the relative order of the columns. :param directory: the directory of the dataset
    :return:
    """
    # get train, validation and test data

    train_data = get_numpy_dataset(os.path.join(directory, 'dom-full-train-*.csv'))
    validation_data = get_numpy_dataset(os.path.join(directory, 'dom-full-validation-*.csv'))
    test_data = get_numpy_dataset(os.path.join(directory, 'dom-full-test-*.csv'))

    # get X and y
    train_X, train_y = train_data['X'], train_data['y']
    validation_X, validation_y = validation_data['X'], validation_data['y']
    test_X, test_y = test_data['X'], test_data['y']

    # get the split indices - then build the slice objects
    split_points = np.cumsum([0, train_y.size, validation_y.size, test_y.size])
    split_slices = [slice(i, j) for i, j in zip(split_points[:-1], split_points[1:])]
    return np.concatenate((train_X, validation_X, test_X)), np.concatenate(
        (train_y, validation_y, test_y)), split_slices
