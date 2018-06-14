import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import clone
from sklearn.base import BaseEstimator, TransformerMixin


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
        all_args = dict(filter(lambda item: item[1] is not None, all_args.items()))

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