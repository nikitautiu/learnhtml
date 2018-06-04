from unittest import TestCase

from utils import zip_dicts, dict_combinations


class TestUtils(TestCase):
    def test_zip_dicts(self):
        dict_list = [{1: 1, 2: 3}, {2: 1, 1: 2}, {1: 3, 2: 7}]
        result = zip_dicts(*dict_list)
        expected_result = {1: [1, 2, 3], 2: [3, 1, 7]}

        self.assertDictEqual(result, expected_result)

    def test_dict_combinations(self):
        """Test whether dict combinations outputs as expected"""
        lists = [
            [{'a': [1, 2, 3], 'b': [1]}],
            [{'c': [4]}, {'a': [5]}]
        ]
        result = list(dict_combinations(*lists))
        result = 9