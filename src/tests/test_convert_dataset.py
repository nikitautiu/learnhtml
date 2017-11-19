import logging
import unittest

import pandas as pd

from dataset_conversion import lcs
from dataset_conversion.conversion import convert_dataset, get_block_ratios, get_blocks_for_file, get_ratios_per_html, \
    extract_ratios_from_df


class TestConvertDataset(unittest.TestCase):
    @unittest.skip('not yet')
    def convert_dragnet(self):
        # suppress all those annotinh warnings
        logging.getLogger().setLevel(logging.ERROR)

        # try to convert the dataset_dragnet
        htmls, labels = convert_dataset(directory='src/tests/dataset_dragnet',
                                        prefix='dragnet-', cleaneval=False, return_ratios=True)

        labels_578 = labels[labels.url.str.contains('R578.html') & labels.content_label].compute()
        labels_9 = labels[labels.url.str.contains('9.html') & labels.content_label].compute()

        expected_paths578 = {
            '/html/body/div[1]/div/div[7]/div/div[2]/h1',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[1]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[2]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[3]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[4]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[5]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[6]',
        }

        expected_paths9 = {
            '/html/body/div[3]/div[1]/div[1]/div/div/div/h1',
            '/html/body/div[3]/div[1]/div[3]/div[1]',
            '/html/body/div[3]/div[1]/div[3]/div[2]',
            '/html/body/div[3]/div[1]/div[3]/div[3]',
            '/html/body/div[3]/div[1]/div[3]/div[4]/span',
            '/html/body/div[3]/div[1]/p[1]',
            '/html/body/div[3]/div[1]/p[2]',
            '/html/body/div[3]/div[1]/p[3]',
            '/html/body/div[3]/div[1]/p[4]',
            '/html/body/div[3]/div[1]/p[5]',
            '/html/body/div[3]/div[1]/p[6]',
            '/html/body/div[3]/div[1]/p[7]',
            '/html/body/div[3]/div[1]/p[8]',
            '/html/body/div[3]/div[1]/p[9]',
            '/html/body/div[3]/div[1]/p[10]',
            '/html/body/div[3]/div[1]/p[11]',
        }
        self.assertSetEqual(set(labels_578.path), expected_paths578)
        self.assertSetEqual(set(labels_9.path), expected_paths9)

        # try to convert the dataset_dragnet
        htmls, labels = convert_dataset(directory='src/tests/dataset_cleaneval',
                                        prefix='cleaneval-', cleaneval=True)

        labels_1 = labels[labels.url.str.contains('1.html') & labels.content_label].compute()
        labels_2 = labels[labels.url.str.contains('2.html') & labels.content_label].compute()

        expected_paths1 = {
            '/html/body/text/div/center/table/tr/td[2]/table/tr[3]/td/p[1]',
            '/html/body/text/div/center/table/tr/td[2]/table/tr[3]/td/p[1]/b',
            '/html/body/text/div/center/table/tr/td[2]/table/tr[3]/td/ul/li[1]',
            '/html/body/text/div/center/table/tr/td[2]/table/tr[3]/td/ul/li[1]/u',
            '/html/body/text/div/center/table/tr/td[2]/table/tr[3]/td/ul/li[1]/u/font',
            '/html/body/text/div/center/table/tr/td[2]/table/tr[3]/td/ul/li[2]',
            '/html/body/text/div/center/table/tr/td[2]/table/tr[3]/td/ul/li[2]/font',
            '/html/body/text/div/center/table/tr/td[2]/table/tr[3]/td/div/center/table/tr[1]',
            '/html/body/text/div/center/table/tr/td[2]/table/tr[3]/td/div/center/table/tr[1]/td',
            '/html/body/text/div/center/table/tr/td[2]/table/tr[3]/td/div/center/table/tr[1]/td/p[1]',
            '/html/body/text/div/center/table/tr/td[2]/table/tr[3]/td/div/center/table/tr[1]/td/p[1]/b',
            '/html/body/text/div/center/table/tr/td[2]/table/tr[3]/td/div/center/table/tr[1]/td/p[2]',
            '/html/body/text/div/center/table/tr/td[2]/table/tr[3]/td/div/center/table/tr[1]/td/p[2]/b',
            '/html/body/text/div/center/table/tr/td[2]/table/tr[3]/td/div/center/table/tr[1]/td/p[3]',
            '/html/body/text/div/center/table/tr/td[2]/table/tr[3]/td/div/center/table/tr[1]/td/p[3]/b',
            '/html/body/text/div/center/table/tr/td[2]/table/tr[3]/td/div/center/table/tr[3]/td/p[2]',
            '/html/body/text/div/center/table/tr/td[2]/table/tr[3]/td/div/center/table/tr[3]/td/p[3]'
        }
        expected_paths2 = {
            '/html/body/text/table[3]/tr/td/h3[1]',
            '/html/body/text/table[3]/tr/td/h3[1]/font',
            '/html/body/text/table[3]/tr/td/h3[2]',
            '/html/body/text/table[3]/tr/td/h3[2]/font',
            '/html/body/text/table[3]/tr/td/p[9]',
            '/html/body/text/table[3]/tr/td/p[9]/font',
            '/html/body/text/table[3]/tr/td/p[9]/font/b',
            '/html/body/text/table[3]/tr/td/p[12]',
            '/html/body/text/table[3]/tr/td/p[12]/font',
            '/html/body/text/table[3]/tr/td/p[12]/font/b',
            '/html/body/text/table[3]/tr/td/p[14]',
            '/html/body/text/table[3]/tr/td/p[14]/font',
            '/html/body/text/table[3]/tr/td/p[14]/font/b',
            '/html/body/text/table[3]/tr/td/p[16]',
            '/html/body/text/table[3]/tr/td/p[16]/font',
            '/html/body/text/table[3]/tr/td/p[16]/font/b',
            '/html/body/text/table[3]/tr/td/p[18]',
            '/html/body/text/table[3]/tr/td/p[18]/font',
            '/html/body/text/table[3]/tr/td/p[18]/font/b',
            '/html/body/text/table[3]/tr/td/h4[2]',
            '/html/body/text/table[3]/tr/td/h4[2]/font[2]',
            '/html/body/text/table[3]/tr/td/h4[3]',
            '/html/body/text/table[3]/tr/td/h4[3]/font',
        }
        self.assertSetEqual(set(labels_1.path), expected_paths1)
        self.assertSetEqual(set(labels_2.path), expected_paths2)

    def test_get_block_ratios(self):
        """Tests the function that returns the percentage of
        tokens from all extracted blocks which are from the gold standard"""

        # ger golden standard blocks
        gold_blocks = get_blocks_for_file('R578.html', 'src/tests/dataset_dragnet', cleaneval=False)
        with open('src/tests/dataset_dragnet/HTML/R578.html') as f:
            html = f.read()

        ratios = get_block_ratios(html, gold_blocks)
        expected_ratios = [('/html', 0.0), ('/html/body/div[1]/div/div[1]/div/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[1]/div/div[2]', 0.0), ('/html/body/div[1]/div/div[3]/div', 0.0),
                           ('/html/body/div[1]/div/div[3]/div/div[2]', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[1]/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[1]/div[1]/div/div/div/div/h3', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[1]/div[1]/div/div/div/div/div', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[1]/div[2]', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[1]/div[3]', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[1]/div[3]/div/div/div[1]/h3', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[1]/div[3]/div/div/div[1]/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[1]/div[3]/div/div/div[1]/div[1]/div/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[1]/div[3]/div/div/div[1]/div[1]/div/div[2]', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[1]/div[3]/div/div/div[1]/div[2]', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[1]/div[3]/div/div/div[1]/div[2]/div/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[1]/div[3]/div/div/div[1]/div[2]/div/div[2]', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[1]/div[3]/div/div/div[1]/div[3]', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[1]/div[3]/div/div/div[1]/div[3]/div/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[1]/div[3]/div/div/div[1]/div[3]/div/div[2]', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[1]/div[3]/div/div/div[2]/h3', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[1]/div[4]', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[1]/div[5]', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[2]/h2', 0.0),
                           ('/html/body/div[1]/div/div[4]/div[2]/div/form/div', 0.0),
                           ('/html/body/div[1]/div/div[6]/div/div/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[6]/div/div/div[3]', 0.0),
                           ('/html/body/div[1]/div/div[6]/div/div/div[5]', 0.0),
                           ('/html/body/div[1]/div/div[6]/div/div/div[7]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[1]/div/div[2]/ul/li[1]/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[1]/div/div[2]/ul/li[1]/div[2]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[1]/div/div[2]/ul/li[1]/div[3]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[1]/div/div[2]/ul/li[2]/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[1]/div/div[2]/ul/li[2]/div[2]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[1]/div/div[2]/ul/li[2]/div[3]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[1]/div/div[2]/ul/li[3]/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[1]/div/div[2]/ul/li[3]/div[2]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[1]/div/div[2]/ul/li[3]/div[3]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/h1', 0.5),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[1]/div[1]/div[1]/div', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[1]/div[1]/div[1]/div/div', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[1]/div[1]/div[1]/div/div/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[1]/div[1]/div[1]/div/div/div[1]/div',
                            0.041666666666666664),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[1]/div[1]/div[1]/div/div/div[2]/div[1]',
                            0.0625),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[1]/div[1]/div[1]/div/div/div[2]/div[2]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[1]/div[1]/div[2]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[1]/div[1]/div[3]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[1]/div[1]/div[3]/div/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[1]/div[4]/h2', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[1]', 1.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[2]', 1.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[3]', 1.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[4]', 1.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[5]', 1.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[6]', 1.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[3]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[4]/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[4]/div[1]/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[4]/div[2]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[4]/div[2]/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[5]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[6]/div[1]/div', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[6]/div[2]/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[6]/div[2]/div[3]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[6]/div[2]/div[3]/form/div', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[7]/h2', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[7]/ul/li[1]/h5', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[7]/ul/li[2]/h5', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[7]/ul/li[3]/h5', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[7]/ul/li[4]/h5', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[8]/div[1]/div', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[8]/div[1]/h3', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[8]/div[2]/div[2]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[8]/div[2]/div[2]/form/ul/li[1]/div', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[8]/div[3]/div', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[2]/div[8]/div[3]/h3', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[1]/form/div', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[3]/div/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[3]/div/div[3]/h2', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[3]/div/div[5]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[4]/h3', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[4]/div[1]/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[4]/div[1]/div[2]/div', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[4]/div[2]/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[4]/div[2]/div[2]/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[5]/h3', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[5]/ul/li[5]/h3', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[5]/ul/li[5]/div/div[1]/div/div', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[5]/ul/li[5]/div/div[2]/div[2]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[5]/ul/li[5]/div/div[3]/div', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[7]/h2', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[7]/div[2]/div[1]/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[7]/div[2]/div[2]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[7]/div[3]/div[1]/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[7]/div[3]/div[2]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[7]/div[4]/div[1]/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[7]/div[4]/div[1]/div[2]/div', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[7]/div[4]/div[2]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[7]/div[5]/div[1]/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[7]/div[5]/div[1]/div[2]/div', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[7]/div[5]/div[2]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[7]/div[6]/div[1]/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[7]/div[6]/div[2]', 0.0),
                           ('/html/body/div[1]/div/div[7]/div/div[3]/div/div[7]/div[7]', 0.0),
                           ('/html/body/div[1]/div/div[8]/div[1]', 0.0),
                           ('/html/body/div[1]/div/div[8]/div[2]/form/div', 0.0),
                           ('/html/body/div[1]/div/div[8]/div[3]', 0.0), ('/html/body/div[1]/div/div[8]/div[4]', 0.0),
                           ('/html/body/div[1]/div/div[9]/h3', 0.0), ('/html/body/div[1]/div/div[9]/div', 0.0),
                           ('/html/body/div[1]/div/div[10]', 0.0),
                           ('/html/body/div[1]/div/div[11]/div[2]/ul/li[1]/h4', 0.0),
                           ('/html/body/div[1]/div/div[11]/div[2]/ul/li[2]/h4', 0.0),
                           ('/html/body/div[1]/div/div[11]/div[2]/ul/li[3]/h4', 0.0),
                           ('/html/body/div[1]/div/div[11]/div[2]/ul/li[4]/h4', 0.0),
                           ('/html/body/div[1]/div/div[11]/div[3]/ul/li[1]/h4', 0.0),
                           ('/html/body/div[1]/div/div[11]/div[3]/ul/li[2]/h4', 0.0),
                           ('/html/body/div[1]/div/div[11]/div[3]/ul/li[3]/h4', 0.0),
                           ('/html/body/div[1]/div/div[11]/div[3]/ul/li[4]/h4', 0.0),
                           ('/html/body/div[1]/div/div[11]/div[4]/ul/li[1]/h4', 0.0),
                           ('/html/body/div[1]/div/div[11]/div[5]/ul/li/h4', 0.0),
                           ('/html/body/div[1]/div/div[11]/div[5]/div', 0.0),
                           ('/html/body/div[1]/div/div[11]/div[6]', 0.0), ('/html/body/div[1]/div/div[12]', 0.0),
                           ('/html/body/div[2]', 0.0)]
        self.assertListEqual(expected_ratios, ratios)

        # do for cleaneval
        gold_blocks = get_blocks_for_file('2.html', 'src/tests/dataset_cleaneval', cleaneval=False)
        with open('src/tests/dataset_cleaneval/HTML/2.html') as f:
            html = f.read()

        ratios = get_block_ratios(html, gold_blocks)
        expected_ratios = [
            ('/html/body/text/table[3]/tr/td/h3[1]', 1.0), ('/html/body/text/table[3]/tr/td/h3[2]', 1.0),
            ('/html/body/text/table[3]/tr/td/p[2]', 1.0), ('/html/body/text/table[3]/tr/td/p[3]', 1.0),
            ('/html/body/text/table[3]/tr/td/p[4]', 1.0), ('/html/body/text/table[3]/tr/td/p[5]', 1.0),
            ('/html/body/text/table[3]/tr/td/p[6]', 1.0), ('/html/body/text/table[3]/tr/td/p[7]', 1.0),
            ('/html/body/text/table[3]/tr/td/h4[1]', 1.0), ('/html/body/text/table[3]/tr/td/p[8]', 1.0),
            ('/html/body/text/table[3]/tr/td/p[9]', 1.0),
            ('/html/body/text/table[3]/tr/td/p[10]', 0.9672131147540983),
            ('/html/body/text/table[3]/tr/td/p[11]', 0.9142857142857143),
            ('/html/body/text/table[3]/tr/td/p[12]', 1.0),
            ('/html/body/text/table[3]/tr/td/p[13]', 0.9907407407407407),
            ('/html/body/text/table[3]/tr/td/p[14]', 1.0),
            ('/html/body/text/table[3]/tr/td/p[15]', 0.9929577464788732),
            ('/html/body/text/table[3]/tr/td/p[16]', 1.0), ('/html/body/text/table[3]/tr/td/p[17]', 1.0),
            ('/html/body/text/table[3]/tr/td/p[18]', 1.0),
            ('/html/body/text/table[3]/tr/td/p[19]', 0.9880952380952381),
            ('/html/body/text/table[3]/tr/td/h4[2]', 1.0),
            ('/html/body/text/table[3]/tr/td/p[20]', 0.9921259842519685),
            ('/html/body/text/table[3]/tr/td/p[21]', 1.0),
            ('/html/body/text/table[3]/tr/td/p[22]', 0.9682539682539683),
            ('/html/body/text/table[3]/tr/td/p[23]', 0.9710144927536232),
            ('/html/body/text/table[3]/tr/td/p[24]', 1.0), ('/html/body/text/table[3]/tr/td/p[25]', 1.0),
            ('/html/body/text/table[3]/tr/td/h4[3]', 1.0),
            ('/html/body/text/table[3]/tr/td/p[26]', 0.9245283018867925),
            ('/html/body/text/table[3]/tr/td/p[27]', 0.9069767441860465),
            ('/html/body/text/table[4]', 0.0)
        ]
        self.assertListEqual(expected_ratios, ratios)

    def test_get_blocks_for_file(self):
        """Test golden standard block extraction"""

        # dragnet
        extracted_blocks = get_blocks_for_file('R578.html', 'src/tests/dataset_dragnet', cleaneval=False)
        expected_blocks = [
            '2013 Ford Escape Video Road Test',
            'The Ford Escape is new for 2013, and this time, it\'s no carryover crossover SUV. It\'s '
            'more like a sporty hatchback with great handling and turbo power. But is that what you '
            'need? We take a look in our latest video road test.',
            'Take one look at the new Escape and you\'d hardly recognize it. Gone is the boxy '
            'miniature Explorer styling, replaced by a tightly fitted form that looks part hatchback '
            'and part running shoe.',
            'The Escape\'s dash is swoopy and daring, but as a result, the interior isn\'t as open or '
            'airy as before. We like the clean gauge cluster, and the piano-style controls at the '
            'center stack.',
            'The new Escape is quick and nimble--as long as you pick the right engine. The base '
            '2.5-liter four-cylinder has 155 horsepower; it\'s mostly for rental fleets. There\'s a '
            'turbo 1.6-liter four with 178 horsepower, and a 2.0-liter turbo four with 240 horsepower '
            'like this version.',
            'The new Escape puts gas mileage as a priority, too. The base engine\'s rated at 22 mpg '
            'city and 31 mpg highway. The 1.6-liter, 24 mpg city and 33 mpg highway. Our 2.0-liter '
            'Escape gets EPA ratings of 22 mpg city and 30 mpg highway. Keep in mind, those numbers '
            'are for front-drive models; all-wheel drive will cut gas mileage by one or two miles per '
            'gallon.',
            'The trade-off for the Escape\'s great handling is a feeling of interior room. The '
            'Escape\'s technically a subcompact, like the 2013 Hyundai Tucson and 2013 Mazda CX-5, '
            'so there\'s just less space than in some bigger crossovers.',
            'There\'s not much knee or foot room, and no Escape offers a power front passenger seat. '
            'In back, there\'s better passenger space, and in all five seats, headroom is very good. '
            'The rear bench seat backs recline for long-distance driving. And the seats fold flat to '
            'create 68 cubic feet of cargo space--just a little less than the room in a 2013 Honda '
            'CR-V.',
            'Safety is near the top of its class, but a bit behind the Honda, though. The Escape\'s '
            'been named a Top Safety Pick by the IIHS, and earned four stars from NHTSA. Ford\'s MyKey '
            'system lets parents limit stereo volume and vehicle speed when handing the keys over to '
            'teen drivers--but Bluetooth and a rearview camera are options on the base models.',
            'The Escape is priced from the mid $20,000s and can be optioned all the way to $38,'
            '000. Standard features include a USB port, a six-speaker audio system, climate control '
            'and power features. The $26,000 Escape SE gets Bluetooth and Ford\'s SYNC voice-control '
            'system. The pricey Titanium model gets the controversial MyFord Touch infotainment '
            'system, leather seats, push button start, and a premium Sony audio sound system. Our '
            'Titanium test vehicle has a sticker price of nearly $35,000.',
            'So what\'s the bottom line with the new Ford Escape? The fake SUV pitch is gone, '
            'and the Escape\'s now a real athlete.',
            'For more information on the 2013 Ford Escape be sure you read our full review here.'
        ]

        self.assertListEqual(expected_blocks, extracted_blocks)

        # cleaneval
        extracted_blocks = get_blocks_for_file('1.html', 'src/tests/dataset_cleaneval', cleaneval=True)
        expected_blocks = [
            'If you feel it\'s time to start taking action and get your property sold FAST and for TOP DOLLAR; '
            'here\'s a good place to start.',
            'Your Home Will Sell', '* Fast', 'For Top Dollar',
            '* With the Least Amount of Hassle You will get these results because of our:', '1. Unique Team System',
            '2. Innovative Consumer Programs', '3. Leading Edge Technology', '4. Specialized Knowledge',
            '72% of homeowners are dissatisfied with their agent\'s performance', 'Why?',
            'The Major Reason: Poor Communication', "The ABC's of Real Estate Marketing (What Most Realtors Do)",
            'Advertise themselves', 'Bang a sign into your lawn', 'Create an ad for the paper (and maybe run it)',
            'Download your listing to the MLS', 'Encourage their office to show it',
            'Figure they might try an open house', 'Get on their knees and pray it will sell',
            'This is the way real estate has been practiced for the past 100 years, and it\'s still the way many '
            'agents operate today, but...',
            '... these traditional methods have proven to be less and less effective.',
            'That\'s why we use the latest technology and proven consumer innovations which go far beyond this '
            'antiquated ABC approach.'
        ]

        self.assertListEqual(expected_blocks, extracted_blocks)

    def test_get_ratios_per_html(self):
        """Test to see if the dataframe is properly returned"""
        gold_blocks = get_blocks_for_file('R578.html', 'src/tests/dataset_dragnet', cleaneval=False)
        with open('src/tests/dataset_dragnet/HTML/R578.html') as f:
            html = f.read()

        df = get_ratios_per_html(html, gold_blocks)
        non_zero_paths = df[df['ratio'] != 0]['path'].tolist()
        expected_result = [
            '/html/body/div[1]/div/div[7]/div/div[2]/h1',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/div[1]/div[1]/div/div/div[1]/div',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/div[1]/div[1]/div/div/div[2]/div[1]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[1]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[2]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[3]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[4]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[5]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[6]',
        ]

        # check to see if properly matched the non zero ones
        self.assertListEqual(non_zero_paths, expected_result)

        # same for cleaneval
        gold_blocks = get_blocks_for_file('2.html', 'src/tests/dataset_cleaneval', cleaneval=True)
        with open('src/tests/dataset_cleaneval/HTML/2.html') as f:
            html = f.read()

        df = get_ratios_per_html(html, gold_blocks)
        non_zero_paths = df[df['ratio'] != 0]['path'].tolist()
        expected_result = [
            '/html/body/text/table[3]/tr/td/h3[1]',
            '/html/body/text/table[3]/tr/td/h3[2]',
            '/html/body/text/table[3]/tr/td/p[2]',
            '/html/body/text/table[3]/tr/td/p[3]',
            '/html/body/text/table[3]/tr/td/p[4]',
            '/html/body/text/table[3]/tr/td/p[5]',
            '/html/body/text/table[3]/tr/td/p[6]',
            '/html/body/text/table[3]/tr/td/p[7]',
            '/html/body/text/table[3]/tr/td/h4[1]',
            '/html/body/text/table[3]/tr/td/p[8]',
            '/html/body/text/table[3]/tr/td/p[9]',
            '/html/body/text/table[3]/tr/td/p[10]',
            '/html/body/text/table[3]/tr/td/p[11]',
            '/html/body/text/table[3]/tr/td/p[12]',
            '/html/body/text/table[3]/tr/td/p[13]',
            '/html/body/text/table[3]/tr/td/p[14]',
            '/html/body/text/table[3]/tr/td/p[15]',
            '/html/body/text/table[3]/tr/td/p[16]',
            '/html/body/text/table[3]/tr/td/p[17]',
            '/html/body/text/table[3]/tr/td/p[18]',
            '/html/body/text/table[3]/tr/td/p[19]',
            '/html/body/text/table[3]/tr/td/h4[2]',
            '/html/body/text/table[3]/tr/td/p[20]',
            '/html/body/text/table[3]/tr/td/p[21]',
            '/html/body/text/table[3]/tr/td/p[22]',
            '/html/body/text/table[3]/tr/td/p[23]',
            '/html/body/text/table[3]/tr/td/p[24]',
            '/html/body/text/table[3]/tr/td/p[25]',
            '/html/body/text/table[3]/tr/td/h4[3]',
            '/html/body/text/table[3]/tr/td/p[26]',
            '/html/body/text/table[3]/tr/td/p[27]'
        ]

        # check to see if properly matched the non zero ones
        self.assertListEqual(non_zero_paths, expected_result)

    def test_extract_ratios_from_df(self):
        # cleaneval
        gold_blocks1 = get_blocks_for_file('2.html', 'src/tests/dataset_cleaneval', cleaneval=True)
        with open('src/tests/dataset_cleaneval/HTML/2.html') as f:
            html1 = f.read()

        # dragnet
        gold_blocks2 = get_blocks_for_file('R578.html', 'src/tests/dataset_dragnet', cleaneval=False)
        with open('src/tests/dataset_dragnet/HTML/R578.html') as f:
            html2 = f.read()

        html_df = pd.DataFrame(data={'url': ['2.html', 'R578.html'],
                                     'html': [html1, html2],
                                     'gold_standard': [gold_blocks1, gold_blocks2]})

        result = extract_ratios_from_df(html_df)
        # check for the dragnet one
        expected_result = [
            '/html/body/div[1]/div/div[7]/div/div[2]/h1',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/div[1]/div[1]/div/div/div[1]/div',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/div[1]/div[1]/div/div/div[2]/div[1]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[1]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[2]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[3]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[4]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[5]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[6]'
        ]
        self.assertListEqual(result[(result['url'] == 'R578.html') & (result['ratio'] != 0)]['path'].tolist(),
                             expected_result)

    def test_convert_dataset(self):
        """Test the parallellized version of the ratio extractor"""
        cleaneval_html, cleaneval_labels = convert_dataset('src/tests/dataset_cleaneval', 'cleaneval-', cleaneval=True,
                                                           return_ratios=True)

        # check urls
        self.assertSetEqual(set(cleaneval_html['url'].unique().compute()),
                            {'file://cleaneval-1.html', 'file://cleaneval-2.html'})

        # print(len(cleaneval_labels[cleaneval_labels.content_label]).compute()
        expected_result = [
            '/html/body/text/table[3]/tr/td/h3[1]',
            '/html/body/text/table[3]/tr/td/h3[2]',
            '/html/body/text/table[3]/tr/td/p[2]',
            '/html/body/text/table[3]/tr/td/p[3]',
            '/html/body/text/table[3]/tr/td/p[4]',
            '/html/body/text/table[3]/tr/td/p[5]',
            '/html/body/text/table[3]/tr/td/p[6]',
            '/html/body/text/table[3]/tr/td/p[7]',
            '/html/body/text/table[3]/tr/td/h4[1]',
            '/html/body/text/table[3]/tr/td/p[8]',
            '/html/body/text/table[3]/tr/td/p[9]',
            '/html/body/text/table[3]/tr/td/p[10]',
            '/html/body/text/table[3]/tr/td/p[11]',
            '/html/body/text/table[3]/tr/td/p[12]',
            '/html/body/text/table[3]/tr/td/p[13]',
            '/html/body/text/table[3]/tr/td/p[14]',
            '/html/body/text/table[3]/tr/td/p[15]',
            '/html/body/text/table[3]/tr/td/p[16]',
            '/html/body/text/table[3]/tr/td/p[17]',
            '/html/body/text/table[3]/tr/td/p[18]',
            '/html/body/text/table[3]/tr/td/p[19]',
            '/html/body/text/table[3]/tr/td/h4[2]',
            '/html/body/text/table[3]/tr/td/p[20]',
            '/html/body/text/table[3]/tr/td/p[21]',
            '/html/body/text/table[3]/tr/td/p[22]',
            '/html/body/text/table[3]/tr/td/p[23]',
            '/html/body/text/table[3]/tr/td/p[24]',
            '/html/body/text/table[3]/tr/td/p[25]',
            '/html/body/text/table[3]/tr/td/h4[3]',
            '/html/body/text/table[3]/tr/td/p[26]',
            '/html/body/text/table[3]/tr/td/p[27]'
        ]
        self.assertListEqual(cleaneval_labels[cleaneval_labels['content_label'] & (
            cleaneval_labels['url'].str.contains('2.html'))]['path'].compute().tolist(), expected_result)

        # now dragnet
        dragnet_html, dragnet_labels = convert_dataset('src/tests/dataset_dragnet', 'dragnet-', cleaneval=False)

        # check urls
        self.assertSetEqual(set(dragnet_html['url'].unique().compute()),
                            {'file://dragnet-9.html', 'file://dragnet-R578.html'})

        # might notice that two are missing unlike in other tests
        # this is because they are under the 0.1 threshold and therefore
        # not included
        expected_result = {
            '/html/body/div[1]/div/div[7]/div/div[2]/h1',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[1]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[2]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[3]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[4]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[5]',
            '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[6]'
        }
        self.assertSetEqual(set(dragnet_labels[dragnet_labels['content_label'] & (
            dragnet_labels['url'].str.contains('R578.html'))]['path'].compute().tolist()), expected_result)

    def test_lcs(self):
        """Test longest common subsequence"""
        seq_a = [b'a', b'b', b'c', b'd']
        seq_b = [b'a', b'b', b'e', b'e', b'c', b'e']

        inclusions = lcs.check_inclusion(seq_a, seq_b)
        expected = [True, True, True, False]
        self.assertListEqual(inclusions, expected)
