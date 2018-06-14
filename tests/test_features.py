from unittest import TestCase

from learnhtml.features import extract_features_from_html


class TestFeatures(TestCase):
    def setUp(self):
        self.html = """
                <html>
                <head>
                    <title>Sompage</title>
                    <script>fdsfsfsdhsfbsdsdhgsdgsdugsduigsdiugdius</script>
                </head>
                <body>

                <h2>An unordered HTML list</h2>

                <ul>
                  <li><a class="aaaa bbs aa">Coffee</a> ffdsdfssdggsd sss <a>salut</a>
                    <p> sadsdasda <a id="cevceva"> asdasdsa </p>
                  </li>
                  <li class="123 ac">Tea</li>
                  <li>Milk</li>
                </ul>  

                </body>
                </html>
                """

    def test_extract_tags(self):
        feats = extract_features_from_html(self.html, 2, 2)
        feats = feats.set_index('path')  # set the index by the path

        # compare
        expected_value = {
            '/html': 'html',
            '/html/body': 'body',
            '/html/body/h2': 'h2',
            '/html/body/ul': 'ul',
            '/html/body/ul/li[1]': 'li',
            '/html/body/ul/li[1]/a[1]': 'a',
            '/html/body/ul/li[1]/a[2]': 'a',
            '/html/body/ul/li[1]/p': 'p',
            '/html/body/ul/li[1]/p/a': 'a',
            '/html/body/ul/li[2]': 'li',
            '/html/body/ul/li[3]': 'li',
            '/html/head': 'head',
            '/html/head/script': 'script',
            '/html/head/title': 'title'
        }
        value = feats['tag'].to_dict()
        self.assertDictEqual(expected_value, value)

    def test_node_features(self):
        """Test whether node-features are extracted properly"""
        feats = extract_features_from_html(self.html, 2, 2)
        feats = feats.set_index('path')  # set the index by the path

        # tags
        expected_values = {
            '/html': 'html',
            '/html/body': 'body',
            '/html/body/h2': 'h2',
            '/html/body/ul': 'ul',
            '/html/body/ul/li[1]': 'li',
            '/html/body/ul/li[1]/a[1]': 'a',
            '/html/body/ul/li[1]/a[2]': 'a',
            '/html/body/ul/li[1]/p': 'p',
            '/html/body/ul/li[1]/p/a': 'a',
            '/html/body/ul/li[2]': 'li',
            '/html/body/ul/li[3]': 'li',
            '/html/head': 'head',
            '/html/head/script': 'script',
            '/html/head/title': 'title'
        }
        values = feats['tag'].to_dict()
        self.assertDictEqual(expected_values, values)

        # depth
        expected_values = {
            '/html': 1,
            '/html/body': 2,
            '/html/body/h2': 3,
            '/html/body/ul': 3,
            '/html/body/ul/li[1]': 4,
            '/html/body/ul/li[1]/a[1]': 5,
            '/html/body/ul/li[1]/a[2]': 5,
            '/html/body/ul/li[1]/p': 5,
            '/html/body/ul/li[1]/p/a': 6,
            '/html/body/ul/li[2]': 4,
            '/html/body/ul/li[3]': 4,
            '/html/head': 2,
            '/html/head/script': 3,
            '/html/head/title': 3
        }
        values = feats['depth'].to_dict()
        self.assertDictEqual(expected_values, values)

        # no_children
        expected_values = {
            '/html': 2,
            '/html/body': 2,
            '/html/body/h2': 0,
            '/html/body/ul': 3,
            '/html/body/ul/li[1]': 3,
            '/html/body/ul/li[1]/a[1]': 0,
            '/html/body/ul/li[1]/a[2]': 0,
            '/html/body/ul/li[1]/p': 1,
            '/html/body/ul/li[1]/p/a': 0,
            '/html/body/ul/li[2]': 0,
            '/html/body/ul/li[3]': 0,
            '/html/head': 2,
            '/html/head/script': 0,
            '/html/head/title': 0
        }
        values = feats['no_children'].to_dict()
        self.assertDictEqual(expected_values, values)

        # sibling_pos
        expected_values = {
            '/html': 0,
            '/html/body': 1,
            '/html/body/h2': 0,
            '/html/body/ul': 1,
            '/html/body/ul/li[1]': 0,
            '/html/body/ul/li[1]/a[1]': 0,
            '/html/body/ul/li[1]/a[2]': 1,
            '/html/body/ul/li[1]/p': 2,
            '/html/body/ul/li[1]/p/a': 0,
            '/html/body/ul/li[2]': 1,
            '/html/body/ul/li[3]': 2,
            '/html/head': 0,
            '/html/head/script': 1,
            '/html/head/title': 0
        }
        values = feats['sibling_pos'].to_dict()
        self.assertDictEqual(expected_values, values)

        # id_len
        expected_values = {
            '/html': 0,
            '/html/body': 0,
            '/html/body/h2': 0,
            '/html/body/ul': 0,
            '/html/body/ul/li[1]': 0,
            '/html/body/ul/li[1]/a[1]': 0,
            '/html/body/ul/li[1]/a[2]': 0,
            '/html/body/ul/li[1]/p': 0,
            '/html/body/ul/li[1]/p/a': 7,
            '/html/body/ul/li[2]': 0,
            '/html/body/ul/li[3]': 0,
            '/html/head': 0,
            '/html/head/script': 0,
            '/html/head/title': 0
        }
        values = feats['id_len'].to_dict()
        self.assertDictEqual(expected_values, values)

        # id_text
        expected_values = {
            '/html': '',
            '/html/body': '',
            '/html/body/h2': '',
            '/html/body/ul': '',
            '/html/body/ul/li[1]': '',
            '/html/body/ul/li[1]/a[1]': '',
            '/html/body/ul/li[1]/a[2]': '',
            '/html/body/ul/li[1]/p': '',
            '/html/body/ul/li[1]/p/a': 'cevceva',
            '/html/body/ul/li[2]': '',
            '/html/body/ul/li[3]': '',
            '/html/head': '',
            '/html/head/script': '',
            '/html/head/title': ''
        }
        values = feats['id_text'].to_dict()
        self.assertDictEqual(expected_values, values)

        # no_classes
        expected_values = {
            '/html': 0,
            '/html/body': 0,
            '/html/body/h2': 0,
            '/html/body/ul': 0,
            '/html/body/ul/li[1]': 0,
            '/html/body/ul/li[1]/a[1]': 3,
            '/html/body/ul/li[1]/a[2]': 0,
            '/html/body/ul/li[1]/p': 0,
            '/html/body/ul/li[1]/p/a': 0,
            '/html/body/ul/li[2]': 2,
            '/html/body/ul/li[3]': 0,
            '/html/head': 0,
            '/html/head/script': 0,
            '/html/head/title': 0
        }
        values = feats['no_classes'].to_dict()
        self.assertDictEqual(expected_values, values)

        # class_len
        expected_values = {
            '/html': 0,
            '/html/body': 0,
            '/html/body/h2': 0,
            '/html/body/ul': 0,
            '/html/body/ul/li[1]': 0,
            '/html/body/ul/li[1]/a[1]': 11,
            '/html/body/ul/li[1]/a[2]': 0,
            '/html/body/ul/li[1]/p': 0,
            '/html/body/ul/li[1]/p/a': 0,
            '/html/body/ul/li[2]': 6,
            '/html/body/ul/li[3]': 0,
            '/html/head': 0,
            '/html/head/script': 0,
            '/html/head/title': 0
        }
        values = feats['class_len'].to_dict()
        self.assertDictEqual(expected_values, values)

        # class_text
        expected_values = {
            '/html': '',
            '/html/body': '',
            '/html/body/h2': '',
            '/html/body/ul': '',
            '/html/body/ul/li[1]': '',
            '/html/body/ul/li[1]/a[1]': 'aaaa bbs aa',
            '/html/body/ul/li[1]/a[2]': '',
            '/html/body/ul/li[1]/p': '',
            '/html/body/ul/li[1]/p/a': '',
            '/html/body/ul/li[2]': '123 ac',
            '/html/body/ul/li[3]': '',
            '/html/head': '',
            '/html/head/script': '',
            '/html/head/title': ''
        }
        values = feats['class_text'].to_dict()
        self.assertDictEqual(expected_values, values)

    def test_ancestor_features1(self):
        """Test first level(parent) ancestor features"""

        feats = extract_features_from_html(self.html, 2, 2)
        feats = feats.set_index('path')  # set the index by the path

        # tags
        expected_values = {
            '/html': '',
            '/html/body': 'html',
            '/html/body/h2': 'body',
            '/html/body/ul': 'body',
            '/html/body/ul/li[1]': 'ul',
            '/html/body/ul/li[1]/a[1]': 'li',
            '/html/body/ul/li[1]/a[2]': 'li',
            '/html/body/ul/li[1]/p': 'li',
            '/html/body/ul/li[1]/p/a': 'p',
            '/html/body/ul/li[2]': 'ul',
            '/html/body/ul/li[3]': 'ul',
            '/html/head': 'html',
            '/html/head/script': 'head',
            '/html/head/title': 'head'
        }
        values = feats['ancestor1_tag'].to_dict()
        self.assertDictEqual(expected_values, values)

        # depth
        expected_values = {
            '/html': 0,
            '/html/body': 1,
            '/html/body/h2': 2,
            '/html/body/ul': 2,
            '/html/body/ul/li[1]': 3,
            '/html/body/ul/li[1]/a[1]': 4,
            '/html/body/ul/li[1]/a[2]': 4,
            '/html/body/ul/li[1]/p': 4,
            '/html/body/ul/li[1]/p/a': 5,
            '/html/body/ul/li[2]': 3,
            '/html/body/ul/li[3]': 3,
            '/html/head': 1,
            '/html/head/script': 2,
            '/html/head/title': 2
        }
        values = feats['ancestor1_depth'].to_dict()
        self.assertDictEqual(expected_values, values)

        # no_children
        expected_values = {
            '/html': 0,
            '/html/body': 2,
            '/html/body/h2': 2,
            '/html/body/ul': 2,
            '/html/body/ul/li[1]': 3,
            '/html/body/ul/li[1]/a[1]': 3,
            '/html/body/ul/li[1]/a[2]': 3,
            '/html/body/ul/li[1]/p': 3,
            '/html/body/ul/li[1]/p/a': 1,
            '/html/body/ul/li[2]': 3,
            '/html/body/ul/li[3]': 3,
            '/html/head': 2,
            '/html/head/script': 2,
            '/html/head/title': 2
        }
        values = feats['ancestor1_no_children'].to_dict()
        self.assertDictEqual(expected_values, values)

        # sibling_pos
        expected_values = {
            '/html': 0,
            '/html/body': 0,
            '/html/body/h2': 1,
            '/html/body/ul': 1,
            '/html/body/ul/li[1]': 1,
            '/html/body/ul/li[1]/a[1]': 0,
            '/html/body/ul/li[1]/a[2]': 0,
            '/html/body/ul/li[1]/p': 0,
            '/html/body/ul/li[1]/p/a': 2,
            '/html/body/ul/li[2]': 1,
            '/html/body/ul/li[3]': 1,
            '/html/head': 0,
            '/html/head/script': 0,
            '/html/head/title': 0
        }
        values = feats['ancestor1_sibling_pos'].to_dict()
        self.assertDictEqual(expected_values, values)

        # id_len
        expected_values = {
            '/html': 0,
            '/html/body': 0,
            '/html/body/h2': 0,
            '/html/body/ul': 0,
            '/html/body/ul/li[1]': 0,
            '/html/body/ul/li[1]/a[1]': 0,
            '/html/body/ul/li[1]/a[2]': 0,
            '/html/body/ul/li[1]/p': 0,
            '/html/body/ul/li[1]/p/a': 0,
            '/html/body/ul/li[2]': 0,
            '/html/body/ul/li[3]': 0,
            '/html/head': 0,
            '/html/head/script': 0,
            '/html/head/title': 0
        }
        values = feats['ancestor1_id_len'].to_dict()
        self.assertDictEqual(expected_values, values)

        # id_text
        expected_values = {
            '/html': '',
            '/html/body': '',
            '/html/body/h2': '',
            '/html/body/ul': '',
            '/html/body/ul/li[1]': '',
            '/html/body/ul/li[1]/a[1]': '',
            '/html/body/ul/li[1]/a[2]': '',
            '/html/body/ul/li[1]/p': '',
            '/html/body/ul/li[1]/p/a': '',
            '/html/body/ul/li[2]': '',
            '/html/body/ul/li[3]': '',
            '/html/head': '',
            '/html/head/script': '',
            '/html/head/title': ''
        }
        values = feats['ancestor1_id_text'].to_dict()
        self.assertDictEqual(expected_values, values)

        # no_classes
        expected_values = {
            '/html': 0,
            '/html/body': 0,
            '/html/body/h2': 0,
            '/html/body/ul': 0,
            '/html/body/ul/li[1]': 0,
            '/html/body/ul/li[1]/a[1]': 0,
            '/html/body/ul/li[1]/a[2]': 0,
            '/html/body/ul/li[1]/p': 0,
            '/html/body/ul/li[1]/p/a': 0,
            '/html/body/ul/li[2]': 0,
            '/html/body/ul/li[3]': 0,
            '/html/head': 0,
            '/html/head/script': 0,
            '/html/head/title': 0
        }
        values = feats['ancestor1_no_classes'].to_dict()
        self.assertDictEqual(expected_values, values)

        # class_len
        expected_values = {
            '/html': 0,
            '/html/body': 0,
            '/html/body/h2': 0,
            '/html/body/ul': 0,
            '/html/body/ul/li[1]': 0,
            '/html/body/ul/li[1]/a[1]': 0,
            '/html/body/ul/li[1]/a[2]': 0,
            '/html/body/ul/li[1]/p': 0,
            '/html/body/ul/li[1]/p/a': 0,
            '/html/body/ul/li[2]': 0,
            '/html/body/ul/li[3]': 0,
            '/html/head': 0,
            '/html/head/script': 0,
            '/html/head/title': 0
        }
        values = feats['ancestor1_class_len'].to_dict()
        self.assertDictEqual(expected_values, values)

        # class_text
        expected_values = {
            '/html': '',
            '/html/body': '',
            '/html/body/h2': '',
            '/html/body/ul': '',
            '/html/body/ul/li[1]': '',
            '/html/body/ul/li[1]/a[1]': '',
            '/html/body/ul/li[1]/a[2]': '',
            '/html/body/ul/li[1]/p': '',
            '/html/body/ul/li[1]/p/a': '',
            '/html/body/ul/li[2]': '',
            '/html/body/ul/li[3]': '',
            '/html/head': '',
            '/html/head/script': '',
            '/html/head/title': ''
        }
        values = feats['ancestor1_class_text'].to_dict()
        self.assertDictEqual(expected_values, values)

    def test_descendant_features1(self):
        """Test the first level of descendant features"""

        # 'descendant1_no_nodes', 'descendant1_no_children_avg',
        # 'descendant1_id_len_avg', 'descendant1_no_classes_avg',
        # 'descendant1_class_len_avg', 'descendant1_text_len_avg',
        # 'descendant1_classes', 'descendant1_ids', 'descendant1_tags',
        # 'descendant2_no_nodes', 'descendant2_no_children_avg'

        feats = extract_features_from_html(self.html, 2, 2)
        feats = feats.set_index('path')  # set the index by the path

        # no_nodes
        expected_values = {
            '/html': 2,
            '/html/head': 2,
            '/html/head/title': 0,
            '/html/head/script': 0,
            '/html/body': 2,
            '/html/body/h2': 0,
            '/html/body/ul': 3,
            '/html/body/ul/li[1]': 3,
            '/html/body/ul/li[1]/a[1]': 0,
            '/html/body/ul/li[1]/a[2]': 0,
            '/html/body/ul/li[1]/p': 1,
            '/html/body/ul/li[1]/p/a': 0,
            '/html/body/ul/li[2]': 0,
            '/html/body/ul/li[3]': 0
        }
        values = feats['descendant1_no_nodes'].to_dict()
        self.assertDictEqual(expected_values, values)

        # no_children_avg
        expected_values = {
            '/html': 2.0,
            '/html/head': 0.0,
            '/html/head/title': 0.0,
            '/html/head/script': 0.0,
            '/html/body': 1.5,
            '/html/body/h2': 0.0,
            '/html/body/ul': 1.0,
            '/html/body/ul/li[1]': 0.3333333333333333,
            '/html/body/ul/li[1]/a[1]': 0.0,
            '/html/body/ul/li[1]/a[2]': 0.0,
            '/html/body/ul/li[1]/p': 0.0,
            '/html/body/ul/li[1]/p/a': 0.0,
            '/html/body/ul/li[2]': 0.0,
            '/html/body/ul/li[3]': 0.0
        }
        values = feats['descendant1_no_children_avg'].to_dict()
        self.assertAlmostEqual(values, expected_values)  # float values, so needed

        # id_len_avg
        expected_values = {
            '/html': 0.0,
            '/html/head': 0.0,
            '/html/head/title': 0.0,
            '/html/head/script': 0.0,
            '/html/body': 0.0,
            '/html/body/h2': 0.0,
            '/html/body/ul': 0.0,
            '/html/body/ul/li[1]': 0.0,
            '/html/body/ul/li[1]/a[1]': 0.0,
            '/html/body/ul/li[1]/a[2]': 0.0,
            '/html/body/ul/li[1]/p': 7.0,
            '/html/body/ul/li[1]/p/a': 0.0,
            '/html/body/ul/li[2]': 0.0,
            '/html/body/ul/li[3]': 0.0
        }
        values = feats['descendant1_id_len_avg'].to_dict()
        self.assertAlmostEqual(values, expected_values)  # float values, so needed

        # no_classes_avg
        expected_values = {
            '/html': 0.0,
            '/html/head': 0.0,
            '/html/head/title': 0.0,
            '/html/head/script': 0.0,
            '/html/body': 0.0,
            '/html/body/h2': 0.0,
            '/html/body/ul': 0.6666666666666666,
            '/html/body/ul/li[1]': 1.0,
            '/html/body/ul/li[1]/a[1]': 0.0,
            '/html/body/ul/li[1]/a[2]': 0.0,
            '/html/body/ul/li[1]/p': 0.0,
            '/html/body/ul/li[1]/p/a': 0.0,
            '/html/body/ul/li[2]': 0.0,
            '/html/body/ul/li[3]': 0.0
        }
        values = feats['descendant1_no_classes_avg'].to_dict()
        self.assertAlmostEqual(values, expected_values)  # float values, so needed

        # class_len_avg
        expected_values = {
            '/html': 0.0,
            '/html/head': 0.0,
            '/html/head/title': 0.0,
            '/html/head/script': 0.0,
            '/html/body': 0.0,
            '/html/body/h2': 0.0,
            '/html/body/ul': 2.0,
            '/html/body/ul/li[1]': 3.6666666666666665,
            '/html/body/ul/li[1]/a[1]': 0.0,
            '/html/body/ul/li[1]/a[2]': 0.0,
            '/html/body/ul/li[1]/p': 0.0,
            '/html/body/ul/li[1]/p/a': 0.0,
            '/html/body/ul/li[2]': 0.0,
            '/html/body/ul/li[3]': 0.0
        }
        values = feats['descendant1_class_len_avg'].to_dict()
        self.assertAlmostEqual(values, expected_values)  # float values, so needed

        # classes
        expected_values = {
            '/html': ',',
            '/html/head': ',',
            '/html/head/title': '',
            '/html/head/script': '',
            '/html/body': ',',
            '/html/body/h2': '',
            '/html/body/ul': ',1 2 3   a c,',
            '/html/body/ul/li[1]': 'a a a a   b b s   a a,,',
            '/html/body/ul/li[1]/a[1]': '',
            '/html/body/ul/li[1]/a[2]': '',
            '/html/body/ul/li[1]/p': '',
            '/html/body/ul/li[1]/p/a': '',
            '/html/body/ul/li[2]': '',
            '/html/body/ul/li[3]': ''
        }
        values = feats['descendant1_classes'].to_dict()
        self.assertAlmostEqual(values, expected_values)  # float values, so needed

        # ids
        expected_values = {
            '/html': ',',
            '/html/head': ',',
            '/html/head/title': '',
            '/html/head/script': '',
            '/html/body': ',',
            '/html/body/h2': '',
            '/html/body/ul': ',,',
            '/html/body/ul/li[1]': ',,',
            '/html/body/ul/li[1]/a[1]': '',
            '/html/body/ul/li[1]/a[2]': '',
            '/html/body/ul/li[1]/p': 'cevceva',
            '/html/body/ul/li[1]/p/a': '',
            '/html/body/ul/li[2]': '',
            '/html/body/ul/li[3]': ''
        }
        values = feats['descendant1_ids'].to_dict()
        self.assertAlmostEqual(values, expected_values)  # float values, so needed

        # tags
        expected_values = {
            '/html': 'head,body',
            '/html/head': 'title,script',
            '/html/head/title': '',
            '/html/head/script': '',
            '/html/body': 'h2,ul',
            '/html/body/h2': '',
            '/html/body/ul': 'li,li,li',
            '/html/body/ul/li[1]': 'a,a,p',
            '/html/body/ul/li[1]/a[1]': '',
            '/html/body/ul/li[1]/a[2]': '',
            '/html/body/ul/li[1]/p': 'a',
            '/html/body/ul/li[1]/p/a': '',
            '/html/body/ul/li[2]': '',
            '/html/body/ul/li[3]': ''
        }
        values = feats['descendant1_tags'].to_dict()
        self.assertAlmostEqual(values, expected_values)  # float values, so needed
