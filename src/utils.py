import functools
import itertools
from urllib.parse import urlparse
from collections import UserDict


import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin


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


def get_random_split(key, proportions):
    """Given a set of keys and the proportions to split, return the random split
    according to those keys. Returns len(proportions) boolean masks for the split"""
    unique_keys = np.unique(key)
    np.random.shuffle(unique_keys)  # in place shuffle

    # get proportional slices on the unique keys
    split_points = np.floor(np.cumsum([0] + proportions) * unique_keys.size).astype(int)
    split_slices = [slice(i, j) for i, j in zip(split_points[:-1], split_points[1:])]

    return [np.isin(key, unique_keys[split_slice]) for split_slice in split_slices]


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]
    
    
class RecDict(UserDict):
    """Dictionary containing arrays that is indexale, both by keys
    and by slices. Used as a lightweight replacement for recarray"""
    # TODO: add a check for lengths
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        return RecDict({k: val[key] for k, val in self.data.items()})

    def __len__(self):
        """Return the length of the elements, jsut fetch the first one"""
        item = next(iter(self.data.values()))
        if isinstance(item, list):
            return len(item)
        return item.shape[0]
    
    @property
    def shape(self):
        # workaround for sklearn
        return (len(self),)
    