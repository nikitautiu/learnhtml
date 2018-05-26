from unittest import TestCase

from features import extract_features_from_html


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
