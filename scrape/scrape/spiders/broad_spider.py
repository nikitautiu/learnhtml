from urllib.parse import urlparse

import scrapy
from scrapy.linkextractors.lxmlhtml import LxmlLinkExtractor
from scrapy.spiders import Rule, CrawlSpider

from scrape.items import PageItem


def get_domain_from_url(url):
    """Returns the fully-qualified domain fo an url."""
    parsed_uri = urlparse(url)
    return parsed_uri.netloc


class HtmlSpider(CrawlSpider):
    name = "html"
    rules = [
        Rule(LxmlLinkExtractor(unique=True), callback='parse_page', follow=True)
    ]

    start_urls = [
        "https://www.olx.ro/",
        "http://www.piata-az.ro/",
        "https://www.okazii.ro/",
        "https://lajumate.ro/",
        "https://www.emag.ro/",
        "https://www.publi24.ro/",
        "http://online.anuntul.ro/"
        "http://www.olx.ro",
        "http://www.lajumate.ro",
        "http://www.publi24.ro",
        "http://www.bzi.ro/anunturi",
        "http://online.anuntul.ro",
        "https://www.nozi.ro",
        "https://brx.ro",
        "https://www.folosit.com",
        "https://www.firstlook.ro",
        "http://anunturi.telegrafonline.ro",
        "http://anunturi.itbox.ro",
        "http://bizcaf.ro",
        "http://www.micapublicitate.ro",
        "http://www.micapublicitate.net",
        "http://www.anunturipublicitare.ro",
        "http://www.stocuri.com",
        "http://www.anunturi-on-line.ro",
        "http://www.plusanunturi.ro",
        "http://www.anunturi66.ro",
        "http://anunturi.memo.ro",
        "http://buluc.ro",
        "http://www.anunturi.biz",
        "http://www.anunturibazar.ro",
        "http://anunturi.netflash.ro",
        "http://www.pzonline.ro",
        "http://bumer.ro/",
        "http://www.anuntul.org",
        "http://www.rasfoieste.ro",
        "http://www.anunturigratuite.net",
        "http://www.anunturigarla.ro",
        "http://www.bizzyo.com",
        "http://www.anunt24h.ro",
        "http://www.imperator.ro",
        "http://www.anunturigen.ro",
        "http://www.tocmai.xyz",
        "http://www.publicitare.ro",
        "http://anunturi-gratuit.ro",
        "http://www.n-anunturi.com",
        "http://www.evrika.ro",
        "http://teajuta.ro",
        "http://www.anunturgent.ro",
        "http://www.infotext.ro",
        "https://www.chilipirim.ro",
        "http://www.anuntulweb.ro",
        "http://www.portalanunt.ro",
        "http://www.anunturigratis.net",
        "http://www.cocosat.ro/",
        "http://www.anunt-online.ro",
        "http://www.anunturi.micportal.ro",
        "http://www.buzzu.ro",
        "http://www.anunturili.ro",
        "http://www.anuntul.biz",
        "http://www.anunturigratuite.ro/",
        "http://www.anunt.net",
        "http://azon.ro/",
        "http://www.anunturi101.ro",
        "http://anunturi360.ro",
        "http://www.buysale.ro",
        "http://anoonturi.ro",
        "http://anunturilocale.eu",
        "http://www.oradea-online.ro/anunturi/",
        "http://mufmuf.ro",
        "http://anunturiexpres.ro",
        "http://www.anunturighid.ro/",
        "http://www.anunturipeinternet.ro",
        "http://www.anuntpromovat.ro",
        "http://anunturi.fullonline.ro",
        "http://www.anunt-gratuit.ro",
        "http://anunturigr.ro",
        "http://www.publicate.ro",
        "http://www.megaanunturi.ro",
        "http://www.anuntulzero.ro",
        "http://anuntulz.ro",
        "http://www.omnianunturi.ro",
        "http://anunturivalabile.ro",
        "http://www.anuntuldeiasi.ro",
        "http://anuntulabc.ro/",
        "http://www.lavedere.ro",
        "http://postezi.ro",
        "http://www.anuntnational.com",
        "http://www.anunturitop.ro",
        "http://www.anunturiazi.ro",
        "http://www.shopconect.ro",
        "http://www.micapublicitate-online.ro",
        "http://www.anunturi-net.ro",
        "http://www.poiu.ro",
        "http://www.e-maz.ro",
        "http://www.etim.ro",
        "http://www.adlist.ro",
        "http://www.infopublicitate.com",
        "http://www.anunturia.ro",
        "http://anuntul10.ro",
        "http://www.anuntgratis.ro",
        "http://www.ecombinatii.ro",
        "http://anuntul-mobil.ro/",
        "http://www.infoanuntul.ro",
        "http://www.anuntgratuit.ro",
        "http://www.anunturidepenet.ro",
        "http://totulaici.ro/",
        "https://vandpeloc.ro",
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