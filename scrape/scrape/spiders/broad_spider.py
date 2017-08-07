from urllib.parse import urlparse

import scrapy
from scrapy.linkextractors.lxmlhtml import LxmlLinkExtractor
from scrapy.spiders import Rule, CrawlSpider

from scrape.items import PageItem


def get_domain_from_url(url):
    """Returns the fully-qualified domain fo an url."""
    return url.split('/')[2]

class HtmlSpider(CrawlSpider):
    name = "html"
    rules = [
        Rule(LxmlLinkExtractor(unique=True), callback='parse_page',
             follow=True, process_request='preprocess_request')
    ]

    start_urls = [
        "https://www.olx.ro/",
        "http://www.piata-az.ro/",
        "https://www.okazii.ro/",
        "https://lajumate.ro/",
        "https://www.emag.ro/",
        "https://www.aliexpress.com",
        "https://www.amazon.com/"
    ]

    def __init__(self, start_url=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # keep the first url, this ensure that we know where we
        # should have started, regardless of redirects
        if start_url is not None:
            if type(start_url) is list:
                self.start_urls = start_url  # list
            else:
                # else treat it like a whitespace separated list of urls
                self.start_urls = start_url.split()

        self.allowed_domains = [get_domain_from_url(url) for url in
                                self.start_urls]

    def parse_page(self, response):
        yield PageItem(url=response.url, html=response.text)

    def preprocess_request(selfself, request):
        """Receives a request and returns the splash equivalent"""

        # it'so nly a matter of setting the meta argument
        request.meta['splash'] = {
            'args': {'html': 1}
        }
        return request