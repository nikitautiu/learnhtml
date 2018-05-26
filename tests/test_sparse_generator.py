from unittest import TestCase

import keras
import numpy as np
from scipy import sparse
from sklearn.datasets import make_classification, make_blobs
from sklearn.metrics import accuracy_score

from keras_utils import sparse_generator, create_model, KerasSparseClassifier


class TestSparseGenerator(TestCase):
    def test_sparse_generator(self):
        # define a sparse matrix
        data_X = [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10]
        ]
        data_y = [
            1, 2, 3, 4, 5
        ]
        arr_X = sparse.csr_matrix(data_X)
        arr_y = np.array(data_y)
        gener = sparse_generator(arr_X, arr_y, 2, shuffle=False)

        # check sequence by sequence
        batch = next(gener)
        self.assertTrue(np.array_equal(batch[0], np.array([[1, 2], [3, 4]])))
        self.assertTrue(np.array_equal(batch[1], np.array([1, 2])))

        batch = next(gener)
        self.assertTrue(np.array_equal(batch[0], np.array([[5, 6], [7, 8]])))
        self.assertTrue(np.array_equal(batch[1], np.array([3, 4])))

        batch = next(gener)
        self.assertTrue(np.array_equal(batch[0], np.array([[9, 10]])))
        self.assertTrue(np.array_equal(batch[1], np.array([5])))

        batch = next(gener)
        self.assertTrue(np.array_equal(batch[0], np.array([[1, 2], [3, 4]])))
        self.assertTrue(np.array_equal(batch[1], np.array([1, 2])))

        # test other shapes
        data_X = [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10]
        ]
        data_y = [
            [1], [2], [3], [4], [5]
        ]
        arr_X = sparse.csr_matrix(data_X)
        arr_y = np.array(data_y)
        gener = sparse_generator(arr_X, arr_y, 2, shuffle=False)

        # check sequence by sequence
        batch = next(gener)
        self.assertTrue(np.array_equal(batch[0], np.array([[1, 2], [3, 4]])))
        self.assertTrue(np.array_equal(batch[1], np.array([[1], [2]])))

        batch = next(gener)
        self.assertTrue(np.array_equal(batch[0], np.array([[5, 6], [7, 8]])))
        self.assertTrue(np.array_equal(batch[1], np.array([[3], [4]])))

        batch = next(gener)
        self.assertTrue(np.array_equal(batch[0], np.array([[9, 10]])))
        self.assertTrue(np.array_equal(batch[1], np.array([[5]])))

        batch = next(gener)
        self.assertTrue(np.array_equal(batch[0], np.array([[1, 2], [3, 4]])))
        self.assertTrue(np.array_equal(batch[1], np.array([[1], [2]])))

    def test_keras_sparse(self):
        X, y = make_blobs(100, n_features=3, centers=2, random_state=0)
        sparse_X, y = sparse.csr_matrix(X), y

        est = KerasSparseClassifier(create_model, nb_features=X.shape[1], batch_size=10, shuffle=True,
                                    optimizer='adam', hidden_layers=[1], activation='sigmoid')

        est.fit(X[:80], y[:80], epochs=300)
        acc_sparse = accuracy_score(y[:80], est.predict(sparse_X[:80]))
        acc = accuracy_score(y[:80], est.predict(X[:80]))
        self.assertEqual(acc_sparse, acc)
        self.assertGreater(acc, .95)

        acc_sparse = accuracy_score(y[80:], est.predict(sparse_X[80:]))
        acc = accuracy_score(y[80:], est.predict(X[80:]))
        self.assertEqual(acc_sparse, acc)
        self.assertGreater(acc, .95)

