"""Classifier package. Contains a general purpose class for HTML classification"""
from learnhtml.features import extract_features_from_html
from learnhtml.model_selection import HeightDepthSelector


class HTMLExtractor(object):
    """HTML extractor class, wraps a normal tag classifier so
    it can receive html and output directly the xpath of the
    extracted tags.

    If depth and height are not specified it will try to extract them
    from the classifier.
    """

    def __init__(self, tag_classifier, depth=None, height=None):
        self.classifier = tag_classifier
        self.depth = depth
        self.height = height

        # try to extract them from the classifier
        if depth is None or height is None:
            verbosity_step = self.classifier.steps[0][1]
            if not isinstance(verbosity_step, HeightDepthSelector):
                raise ValueError('Neither are depth/height specified '
                                 'nor are they present in the classifier')

            verbosity_params = verbosity_step.get_params()
            self.depth = verbosity_params['depth']
            self.height = verbosity_params['height']

    def _get_dataset_from_html(self, html):
        """Returns the df of html given an html document"""
        html_feats = extract_features_from_html(html, height=self.height, depth=self.depth)
        return html_feats

    def extract_from_html(self, html):
        """Return the list of xpaths containing content from the html"""
        html_feats = self._get_dataset_from_html(html)
        features = html_feats.drop(['path'], axis='columns')
        paths = html_feats['path']

        # get the results and convert to list
        results = self.classifier.predict(features)
        return paths.loc[results.astype(bool)].tolist()