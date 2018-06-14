import functools
import itertools
from urllib.parse import urlparse

import numpy as np


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
    a dictionary from each list.

    Basically given a list of lists of dictionaries, returns all the
    dictionaries that are a ersult of sampling items from each of the lists
    """
    combinations = itertools.product(*dict_list_lists)  # cartesian product of all the lists
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


def group_argsort(x, shuffle=True):
    """Sorting function that preserves grouping of elements.
    Same kind elements stay together, but the groups may be shuffled if specified."""
    uniq_elems = np.unique(x)
    if shuffle:
        np.random.shuffle(uniq_elems)

    # get the indices for each unique item
    new_indices = np.zeros(x.shape, dtype=int)
    current_pos = 0
    for elem in uniq_elems:
        elem_indices = np.where(x == elem)[0]  # the positions of elem
        new_indices[current_pos:current_pos + len(elem_indices)] = elem_indices
        current_pos += len(elem_indices)

    return new_indices