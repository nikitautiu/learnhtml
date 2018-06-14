from unittest import TestCase

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.preprocessing import LabelBinarizer

from learnhtml.utils.sklearn import MultiColumnTransformer


class AddTransformer(BaseEstimator, TransformerMixin):
    """Transformer that adds one to every value"""

    def __init__(self, sparse=False):
        self.sparse = sparse

    def transform(self, X, y=None):
        if isinstance(X, pd.Series):
            X = X.values

        if len(X.shape) < 2:
            X = X.reshape((-1, 1))
        if self.sparse:
            return csr_matrix(X + 1)
        return X + 1

    def fit(self, X, y=None):
        pass

    def __eq__(self, other):
        return self.sparse == other.sparse


class TestMultiColumnTransformer(TestCase):
    """Test whether the multicolumn transformer works as intended"""

    def test_transform(self):
        """Test whether transformation works"""
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 2, 2]})
        transformer = MultiColumnTransformer(AddTransformer())

        expected_result = pd.DataFrame({'a': [2, 3, 4], 'b': [3, 3, 3]}).values
        result = transformer.fit_transform(data)
        self.assertTrue(np.array_equal(expected_result, result))

        # test the same but for sparse
        transformer = MultiColumnTransformer(AddTransformer(sparse=True))

        expected_result = pd.DataFrame({'a': [2, 3, 4], 'b': [3, 3, 3]}).values
        result = transformer.fit_transform(data)
        self.assertIsInstance(result, csr_matrix)
        self.assertTrue(np.array_equal(expected_result, result.todense()))

        # test with a conventional transformer
        transformer = MultiColumnTransformer(LabelBinarizer(sparse_output=True))
        data = pd.DataFrame({'a': [1, 2, 1], 'b': [1, 2, 3]})

        expected_result = np.array([[0, 1, 0, 0],
                                    [1, 0, 1, 0],
                                    [0, 0, 0, 1]])
        result = transformer.fit_transform(data)
        self.assertIsInstance(result, csr_matrix)
        self.assertTrue(np.array_equal(expected_result, result.todense()))

        # test with dense
        transformer = MultiColumnTransformer(LabelBinarizer(sparse_output=False))
        data = pd.DataFrame({'a': [1, 2, 1], 'b': [1, 2, 3]})

        expected_result = np.array([[0, 1, 0, 0],
                                    [1, 0, 1, 0],
                                    [0, 0, 0, 1]])
        result = transformer.fit_transform(data)
        self.assertTrue(np.array_equal(expected_result, result))

    def test_params(self):
        """Test whether params are properly implemented"""
        transformer = MultiColumnTransformer(AddTransformer(sparse=False))

        expected_params = {'estimator': AddTransformer()}
        received_params = transformer.get_params(deep=False)
        self.assertDictEqual(expected_params, received_params)

        expected_params = {'estimator': AddTransformer(sparse=False), 'estimator__sparse': False}
        received_params = transformer.get_params(deep=True)
        self.assertDictEqual(expected_params, received_params)

        # set new parameters and test results
        transformer.set_params(estimator=AddTransformer(sparse=True))

        expected_params = {'estimator': AddTransformer(sparse=True), 'estimator__sparse': True}
        received_params = transformer.get_params(deep=True)
        self.assertDictEqual(expected_params, received_params)

        # set only inner params
        transformer.set_params(estimator__sparse=False)

        expected_params = {'estimator': AddTransformer(sparse=False), 'estimator__sparse': False}
        received_params = transformer.get_params(deep=True)
        self.assertDictEqual(expected_params, received_params)

    def test_cloning(self):
        """Test whether the transformer still works after cloning"""
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 2, 2]})
        transformer = MultiColumnTransformer(AddTransformer())

        expected_result = pd.DataFrame({'a': [2, 3, 4], 'b': [3, 3, 3]}).values
        result = transformer.fit_transform(data)
        self.assertTrue(np.array_equal(expected_result, result))

        # clone the transformer
        cloned_transformer = clone(transformer)
        transformer.set_params(estimator__sparse=True)

        result = cloned_transformer.fit_transform(data)
        self.assertIsInstance(result, np.ndarray)