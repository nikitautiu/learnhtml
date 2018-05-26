from unittest import TestCase

from utils import zip_dicts


class TestUtils(TestCase):
    def test_zip_dicts(self):
        dict_list = [{1: 1, 2: 3}, {2: 1, 1: 2}, {1: 3, 2: 7}]
        result = zip_dicts(*dict_list)
        expected_result = {1: [1, 2, 3], 2: [3, 1, 7]}

        self.assertDictEqual(result, expected_result)