import scrapy
from scrapy.linkextractors.lxmlhtml import LxmlLinkExtractor
from scrapy.spiders import CrawlSpider

from scrape.items import PageItem
from utils import get_domain_from_url


class HtmlSpider(CrawlSpider):
    name = "html"

    def __init__(self, start_url=None, follow_links=False, max_pages=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_pages = max_pages

        # keep the first url, this ensure that we know where we
        # should have started, regardless of redirects
        if start_url is not None:
            if type(start_url) is list:
                self.start_urls = start_url  # list
            else:
                # else treat it like a whitespace separated list of urls
                self.start_urls = start_url.split()

        if follow_links:
            # only restrict domains for broad crawl
            self.allowed_domains = {get_domain_from_url(url) for url in self.start_urls}
            # the number of pages crawled for each page
            self.no_of_pages = {domain: 0 for domain in self.allowed_domains}
            # linke extracto
            self.link_extractor = LxmlLinkExtractor(unique=True, allow_domains=self.allowed_domains)

    def parse(self, response):
        """Parse the responses"""
        # increment the number of pages extracted from the page
        yield PageItem(url=response.url, html=response.text)

        # iterate links and traverse
        for link in self.link_extractor.extract_links(response):
            req = self.preprocess_request(scrapy.Request(link.url, self.parse))
            if req:
                yield req

    def start_requests(self):
        """Pass the requests with the appropriate callback"""
        return [self.preprocess_request(req) for req in super().start_requests()]

    def preprocess_request(self, request):
        """Receives a request and returns the splash equivalent"""
        if self.no_of_pages.get(get_domain_from_url(request.url), self.max_pages) > self.max_pages - 1:
            return None  # drop if already crawled enough

        self.no_of_pages[get_domain_from_url(request.url)] += 1
        # it's only a matter of setting the meta argument
        request.meta['splash'] = {
            'args': {'html': 1}
        }
        return request
