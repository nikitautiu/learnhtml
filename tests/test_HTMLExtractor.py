from unittest import TestCase

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from learnhtml.extractor import HTMLExtractor
from learnhtml.model_selection import HeightDepthSelector


class MockClassifier(BaseEstimator, ClassifierMixin):
    def predict(self, X):
        result = np.zeros(shape=(X.shape[0],))
        result[::2] = 1
        return result

    def fit(self, X, y=None):
        pass


class TestHTMLExtractor(TestCase):
    def test_initialization(self):
        """Test whether initialization from a classifier works"""
        cls_pipeline = Pipeline(steps=[('verbosity', HeightDepthSelector(height=1, depth=4)),
                                       ('classify', DummyClassifier(strategy='constant', constant=1))])
        extractor = HTMLExtractor(cls_pipeline)
        self.assertEqual(extractor.height, 1)
        self.assertEqual(extractor.depth, 4)

    def test_classification(self):
        """Test whether classification works as intended"""

        # must prefit the estimators
        height_depth_selector = HeightDepthSelector(height=1, depth=4)
        classifier = MockClassifier()
        height_depth_selector.fit(None)

        cls_pipeline = Pipeline(steps=[('verbosity', height_depth_selector),
                                       ('classify', classifier)])
        extractor = HTMLExtractor(cls_pipeline)
        html_data = '<html><body><p></p></body></html>'

        extract_paths = extractor.extract_from_html(html_data)
        expected_paths = ['/html', '/html/body/p']

        self.assertEqual(len(extract_paths), len(expected_paths))
        self.assertSetEqual(set(extract_paths), set(expected_paths))
