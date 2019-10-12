import logging
from .bools import getters

logger = logging.getLogger(__name__)


def get():
    logger.info("Get method called")
    return getters.get_boolean()
