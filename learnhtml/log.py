import logging

import click_log

logger = logging.getLogger(__name__)

# configure the logger to use the click settings
click_log.basic_config(logger)
