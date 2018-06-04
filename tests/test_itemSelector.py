from unittest import TestCase

import pandas as pd

from utils import ItemSelector


class TestItemSelector(TestCase):
    """Test case for ItemSelector"""

    def test_dataframe(self):
        """Test whether it works with DataFrames"""
        # test series selection
        selector = ItemSelector('a')
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [6, 5, 4], 'c': [1, 0, 1]})

        selected_data = selector.transform(data)
        expected_data = pd.Series([1, 2, 3])
        pd.testing.assert_series_equal(selected_data, expected_data, check_names=False)

        # test dataframe selection
        selector = ItemSelector(['a'])
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [6, 5, 4], 'c': [1, 0, 1]})

        selected_data = selector.transform(data)
        expected_data = pd.DataFrame({'a': [1, 2, 3]})
        pd.testing.assert_frame_equal(selected_data, expected_data)

        # test dataframe selection multiple cols
        selector = ItemSelector(['a', 'c'])
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [6, 5, 4], 'c': [1, 0, 1]})

        selected_data = selector.transform(data)
        expected_data = pd.DataFrame({'a': [1, 2, 3], 'c': [1, 0, 1]})
        pd.testing.assert_frame_equal(selected_data, expected_data)

    def test_dictionary(self):
        """Test whether it works with plain dicts"""
        selector = ItemSelector('a')
        data = {'a': [1, 2, 3], 'b': 6}

        selected_data = selector.transform(data)
        expected_data = [1, 2, 3]
        self.assertListEqual(selected_data, expected_data)

    def test_sparse(self):
        """Test whether it work with sparse DataFrames"""
        # pretty much the same as normal dataframe, but with type checking

        # test series selection
        selector = ItemSelector('a')
        data = pd.SparseDataFrame({'a': [1, 2, 3], 'b': [6, 5, 4], 'c': [1, 0, 1]})

        selected_data = selector.transform(data)
        expected_data = pd.Series([1, 2, 3])
        self.assertIsInstance(selected_data, pd.SparseSeries)  # check datatype
        pd.testing.assert_series_equal(selected_data, expected_data, check_names=False)

        # test dataframe selection
        selector = ItemSelector(['a'])
        data = pd.SparseDataFrame({'a': [1, 2, 3], 'b': [6, 5, 4], 'c': [1, 0, 1]})

        selected_data = selector.transform(data)
        expected_data = pd.SparseDataFrame({'a': [1, 2, 3]})
        self.assertIsInstance(selected_data, pd.SparseDataFrame)  # check
        pd.testing.assert_frame_equal(selected_data, expected_data)

        # test dataframe selection multiple cols
        selector = ItemSelector(['a', 'c'])
        data = pd.SparseDataFrame({'a': [1, 2, 3], 'b': [6, 5, 4], 'c': [1, 0, 1]})

        selected_data = selector.transform(data)
        expected_data = pd.SparseDataFrame({'a': [1, 2, 3], 'c': [1, 0, 1]})
        self.assertIsInstance(selected_data, pd.SparseDataFrame)  # check datatype
        pd.testing.assert_frame_equal(selected_data, expected_data)

    def test_filters(self):
        """Test the DataFrame filtering"""
        data = pd.DataFrame({'a': [1, 2, 3], 'ab': [6, 5, 4], 'c2': [1, 0, 1]})

        # test regex selector
        selector = ItemSelector(regex='[b2]$')
        selected_data = selector.transform(data)
        expected_data = pd.DataFrame({'ab': [6, 5, 4], 'c2': [1, 0, 1]})
        pd.testing.assert_frame_equal(selected_data, expected_data)

        # test items selector
        selector = ItemSelector(items=['a', 'c2'])
        selected_data = selector.transform(data)
        expected_data = pd.DataFrame({'a': [1, 2, 3], 'c2': [1, 0, 1]})
        pd.testing.assert_frame_equal(selected_data, expected_data)

        # test like selector
        selector = ItemSelector(like='a')
        selected_data = selector.transform(data)
        expected_data = pd.DataFrame({'a': [1, 2, 3], 'ab': [6, 5, 4]})
        pd.testing.assert_frame_equal(selected_data, expected_data)

    def test_predicate(self):
        """Test predicate filtering"""
        data = pd.DataFrame({'a': [1, 2, 3], 'ab': ['a', 'ab', 4], 'c2': [1, 0, 1]})

        # name predicate
        selector = ItemSelector(predicate=lambda x: x[0] == 'a')
        selected_data = selector.transform(data)
        expected_data = pd.DataFrame({'a': [1, 2, 3]})
        pd.testing.assert_frame_equal(selected_data, expected_data)

        # type predicate
        selector = ItemSelector(predicate=lambda x: str(x[1]) == 'object')
        selected_data = selector.transform(data)
        expected_data = pd.DataFrame({'ab': ['a', 'ab', 4]})
        pd.testing.assert_frame_equal(selected_data, expected_data)