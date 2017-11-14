from unittest import TestCase

from dataset_conversion import convert_dataset
import logging


class TestConvertDataset(TestCase):
    def test_convert_dragnet(self):
        # suppress all those annotinh warnings
        logging.getLogger().setLevel(logging.ERROR)

        # # try to convert the dataset_dragnet
        # htmls, labels = convert_dataset(directory='src/tests/dataset_dragnet',
        #                                 prefix='dragnet-', cleaneval=False, return_ratios=True)
        #
        # labels_578 = labels[labels.url.str.contains('R578.html') & labels.content_label].compute()
        # labels_9 = labels[labels.url.str.contains('9.html') & labels.content_label].compute()
        #
        # expected_paths578 = {
        #     '/html/body/div[1]/div/div[6]/div/div/div[7]',
        #     '/html/body/div[1]/div/div[6]/div/div/div[7]/span',
        #     '/html/body/div[1]/div/div[7]/div/div[2]/h1',
        #     '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[1]',
        #     '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[2]',
        #     '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[3]',
        #     '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[4]',
        #     '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[5]',
        #     '/html/body/div[1]/div/div[7]/div/div[2]/div[1]/p[6]',
        # }
        #
        # expected_paths9 = {
        #     '/html/body/div[3]/div[1]/div[1]/div/div/div/h1',
        #     '/html/body/div[3]/div[1]/div[3]/div[1]',
        #     '/html/body/div[3]/div[1]/div[3]/div[2]',
        #     '/html/body/div[3]/div[1]/div[3]/div[3]',
        #     '/html/body/div[3]/div[1]/div[3]/div[4]/span',
        #     '/html/body/div[3]/div[1]/p[1]',
        #     '/html/body/div[3]/div[1]/p[2]',
        #     '/html/body/div[3]/div[1]/p[3]',
        #     '/html/body/div[3]/div[1]/p[4]',
        #     '/html/body/div[3]/div[1]/p[5]',
        #     '/html/body/div[3]/div[1]/p[6]',
        #     '/html/body/div[3]/div[1]/p[7]',
        #     '/html/body/div[3]/div[1]/p[8]',
        #     '/html/body/div[3]/div[1]/p[9]',
        #     '/html/body/div[3]/div[1]/p[10]',
        #     '/html/body/div[3]/div[1]/p[11]',
        # }

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