from unittest import TestCase

import pandas as pd

from learnhtml.model_selection import HeightDepthSelector


class TestHeightDepthSelector(TestCase):
    def test_params(self):
        """Test whether params work and internals are not exposed"""
        est = HeightDepthSelector(height=5, depth=4)
        params = est.get_params()
        expected_params = {'height': 5, 'depth': 4}
        self.assertDictEqual(params, expected_params)

        params = est.get_params(deep=True)
        expected_params = {'height': 5, 'depth': 4}
        self.assertDictEqual(params, expected_params)

    def test_selection(self):
        """Test whether selection works"""

        # test dataframe selection multiple cols
        selector = HeightDepthSelector(height=2, depth=0)
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [6, 5, 4], 'ancestor2_aaa': [4, 5, 2],
                             'ancestor3_aaaa':[1, 1, 1], 'descendant7_asdsad': [44, 1, 1]})

        selected_data = selector.fit_transform(data)
        expected_data = pd.DataFrame({'a': [1, 2, 3], 'b': [6, 5, 4], 'ancestor2_aaa': [4, 5, 2]})
        pd.testing.assert_frame_equal(selected_data, expected_data)
