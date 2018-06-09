import functools
import itertools
import tempfile
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from keras.callbacks import Callback
from scipy import sparse
from sklearn import clone
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight

from keras_utils import sparse_generator, KerasSparseClassifier, constrain_memory


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


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """

    def __init__(self, key=None, regex=None, like=None, items=None, predicate=None):
        # do some type checking first
        all_args = {'regex': regex, 'like': like, 'items': items,
                    'predicate': predicate, 'key': key}
        if sum(map(lambda x: x is not None, all_args.values())) > 1:
            raise ValueError('filters are mutually exclusive')
        if sum(map(lambda x: x is not None, all_args.values())) == 0:
            raise ValueError('at least one filter required')

        # save the key and ony the not-null filter
        self.key = key
        self.items = items
        self.regex = regex
        self.like = like
        self.predicate = predicate

    def __repr__(self):
        """Returns the representation of the object"""
        if self.key is not None:
            return 'ItemSelector(key={key})'.format(key=repr(self.key))

        all_args = {'regex': self.regex, 'like': self.like,
                    'items': self.items, 'predicate': self.predicate, 'key': self.key}
        name, key = list(all_args.items())[0]
        return 'ItemSelector({name}={key})'.format(key=repr(key), name=name)

    def set_params(self, key=None, regex=None, like=None, items=None, predicate=None):
        """Sets the parameters of the estimator while also doing a preliminary check"""
        all_args = {'regex': regex, 'like': like, 'items': items,
                    'predicate': predicate, 'key': key}
        if sum(map(lambda x: x is not None, all_args.values())) > 1:
            raise ValueError('filters are mutually exclusive')
        if sum(map(lambda x: x is not None, all_args.values())) == 0:
            raise ValueError('at least one filter required')

        super().set_params(key=key, regex=regex, like=like, items=items, predicate=predicate)

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        # only keep the non-null filter

        filters = {'regex': self.regex, 'like': self.like, 'items': self.items, 'predicate': self.predicate}
        self.filters = dict(filter(lambda item: item[1] is not None, filters.items()))

        if self.key is not None:
            # regardless of type, if key is specified, it should do
            # regular indexing
            return data_dict[self.key]

        if not isinstance(data_dict, (pd.DataFrame, pd.SparseDataFrame)):
            raise ValueError('Only DataFrames can be indexed with filter')

        if self.filters.get('predicate', None) is not None:
            predicate = self.filters.get('predicate')  # use the predicate
            # the predicate receives the column name and dtype

            cols = filter(predicate, zip(data_dict.columns, data_dict.dtypes))
            cols = list(map(lambda x: x[0], cols))
            return data_dict[cols]

        # default to filtering
        return data_dict.filter(**self.filters)


class Metrics(Callback):
    def __init__(self, validation_data, batch_size, *args, prefix='', **kwargs):
        super().__init__(*args, **kwargs)
        self._validation_data = validation_data
        self._batch_size = batch_size
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs={}):
        preds = self.model.predict_generator(
            sparse_generator(self._validation_data[0], None, self._batch_size, shuffle=False),
            steps=np.ceil(self._validation_data[0].shape[0] / self._batch_size)
        )

        predict = np.round(np.asarray(preds))
        target = self._validation_data[1]
        results = {
            'precision': precision_score(target, predict),
            'recall': recall_score(target, predict),
            'f1': f1_score(target, predict)
        }
        print(' - '.join('{}{}: {}'.format(self.prefix, name, val) for name, val in results.items()))

        for name, val in results.items():
            logs['{}{}'.format(self.prefix, name)] = val


class MyKerasClassifier(KerasSparseClassifier):
    """Custom KerasClassifier
    Ensures that we can use early stopping and checkpointing
    """

    def __init__(self, *args, patience=10, expiration=-1, checkpoint_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sk_params['patience'] = patience
        self.sk_params['checkpoint_file'] = checkpoint_file
        self.sk_params['expiration'] = expiration  # number of predicts after which to delete the model

    def fit(self, X, y, **kwargs):
        # local imports, needed for multiprocessing
        import keras
        from keras import backend as K
        constrain_memory()

        # delete model flag. tells the estimator to delete the model after a said number of turns
        # after finishing the first predict after a fit. prevents memory leaks?
        self._predict_turns = 0

        # cleanup the memory. We can't run models in a parallel anyway, so at least, prevent
        # the huge memory leak
        if 'tensorflow' == K.backend():
            K.clear_session()

        # declare the additional kwargs to pass done to the classifier
        additional_sk_params = {}

        # leave a 10% chunk out on which to do validation
        additional_sk_params['validation_data'] = self.sk_params.get('validation_data', None)
        if additional_sk_params['validation_data'] is None:
            val_point = int(X.shape[0] * .9)
            additional_sk_params['validation_data'] = (X[val_point:, :], y[val_point:])
            X = X[:val_point, :]
            y = y[:val_point]

        # try to get the checkpoint file, otherwise use a temporary
        checkpoint_file = self.sk_params.get('checkpoint_file', None)
        is_tmp = False
        if checkpoint_file is None:
            # create a temprorary file to save the checkpoint to
            is_tmp = True
            tmp_file = tempfile.NamedTemporaryFile()
            checkpoint_file = tmp_file.name

        metrics = Metrics(additional_sk_params['validation_data'], 1024, prefix='val_')
        early_stopper = keras.callbacks.EarlyStopping(monitor='val_f1', min_delta=0.0001,
                                                      patience=self.sk_params.get('patience', 10),
                                                      verbose=1, mode='max')
        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_file, monitor='val_f1', verbose=1, save_best_only=True,
                                                     mode='max')

        # set the calbacks per fit method, this ensures that each clone has its own callbacks
        self.sk_params['nb_features'] = X.shape[1]
        self.sk_params['batch_size'] = 1024
        self.sk_params['callbacks'] = [metrics, checkpoint, early_stopper]

        if self.sk_params.get('class_weight', None) == 'balanced':
            weights = class_weight.compute_class_weight('balanced', [0, 1], y)
            additional_sk_params['class_weight'] = dict(enumerate(weights))

        # update the params wih the injected ones
        kwargs.update(additional_sk_params)
        super().fit(X, y, **kwargs)

        # reaload from checkpoint
        self.model.load_weights(checkpoint_file)

        if is_tmp:
            # if it is temporary, delete it at the end
            tmp_file.close()

    def predict(self, *args, **kwargs):
        # TODO: this is just the most horrid workaround ever
        # wrapper that deletes the model if the estimator reaches EXPIRATION
        # the rationale is that after finishing fitting, girdsearch does 2
        # predictions, after which the model remains in memory, causing a memory leak

        result = super().predict(*args, **kwargs)
        self._predict_turns += 1

        if self._predict_turns == self.sk_params['expiration']:
            from keras import backend as K
            if 'tensorflow' == K.backend():
                import tensorflow as tf
                K.clear_session()  # clear the session just for good measure
                tf.reset_default_graph()  # this is needed for the python state
            del self.model

        return result


class MultiColumnTransformer(BaseEstimator, TransformerMixin):
    """Transformer that takes an estimator and applies it to all the columns
    of a DataFrame. Properly handles sparse outputs."""

    def __init__(self, estimator):
        """Creates the transformer. Receives a transformer to wrap."""
        self.estimator = estimator  # estimator to use on the selected columns
        self.cloned_estimators = {}

    def __repr__(self):
        """Return the representation of the estimator."""
        return 'MultiColumnTransformer(estimator={})'.format(repr(self.estimator))

    def fit(self, X, y=None):
        """Fits an estimator for each of the columns in X."""
        self.cloned_estimators = {}

        # clone the estimators for each column and fit
        for col in X.columns.tolist():
            # clone the estimator for every column
            # and fit it to the data
            cloned_est = clone(self.estimator)
            cloned_est.fit(X[col])
            self.cloned_estimators[col] = cloned_est

        return self  # not relevant here

    def transform(self, X, y=None):
        """Transforms every column with its pre-fitted estimator."""

        # transform every given col
        extracted = []
        for col in X.columns.tolist():
            transformed_data = self.cloned_estimators[col].transform(X[col])
            extracted.append(transformed_data)

        if any(sparse.issparse(fea) for fea in extracted):
            stacked = sparse.hstack(extracted).tocsr()
        else:
            stacked = np.hstack(extracted)

        return stacked

