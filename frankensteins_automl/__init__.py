import logging
import logging.config

logging.config.fileConfig("config/logging.conf")

logging.getLogger(__name__).addHandler(logging.NullHandler())
