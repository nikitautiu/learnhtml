import os
import re

import numpy as np
import sparse
from dask import dataframe as dd

from utils import get_random_split


def get_numpy_dataset(csv_pattern, numeric_cols=None, text_cols=None, categorize_id=True, sparse_matrix=True):
    """Given a csv pattern for a dataset, return a dict of ('id', 'numeric', 'text', 'y', 'is_block') containing the components of
    the dataset. `is_block` represents whether the tag corresponds to block as extrcted in the dragnet paper.
    :param text_cols: the columns to select from the textual data
    :param sparse_matrix: whether X should be returned as a COO matrix(greatly reduces memory use)
    :param categorize_id: if True(default), sorts the dataset by the id column, and the column is categorized
    :param numeric_cols: the columns to keep from the numeric data(defaults to all)
    :param csv_pattern: the pattern of the csv files
    :return: the specified dict
    """
    ddf = dd.read_csv(csv_pattern, dtype={'ancestor5_class_text': 'object', 'id_text': 'object'})
    textual_cols = list(filter(lambda col: re.match(r'.*(((id|class)_text)|text)$', col), ddf.columns))

    # separate the numpy arrays
    id_arr = ddf[['url', 'path']].values.compute()
    # numerical features
    numeric_ddf = ddf.drop(['url', 'path', 'content_label', 'is_extracted_block'] + textual_cols, axis=1)
    # text features
    text_ddf = ddf[textual_cols]
    # array that keeps whether that is a block
    extracted_arr = ddf[['is_extracted_block']].values.compute()

    if numeric_cols is not None:
        numeric_ddf = numeric_ddf[numeric_cols]  # keep only the given cols if passed
    if text_cols is not None:
        text_ddf = text_ddf[text_cols]  # keep only the given cols if passed
    text_arr = text_ddf.values.compute()

    y_arr = ddf['content_label'].values.compute()

    # sparsify the matrix if necessary
    if sparse_matrix:
        # return the scipy COO matrix corresponding to the input
        # greatly reduces the input size
        numeric_arr = numeric_ddf.values.map_blocks(sparse.COO).astype('float32').compute().tocsr()
    else:
        numeric_arr = numeric_ddf.values.compute()

    # categorize id if necessary
    if categorize_id:
        _, id_arr = np.unique(id_arr[:, 0], return_inverse=True)

    text_arr[text_arr != text_arr] = ''  # fixes the nan problem
    return {'id': id_arr, 'numeric': numeric_arr,
            'text': text_arr, 'y': y_arr, 'is_block': extracted_arr}


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

