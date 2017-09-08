from urllib.parse import urlparse


def get_domain_from_url(url):
    """Returns the fully-qualified domain fo an url."""
    return urlparse(url).netloc