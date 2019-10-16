import logging


logger = logging.getLogger(__name__)


class ParameterDomain(object):
    def __init__(self, parameter_descriptions):
        self.parameter_descriptions = parameter_descriptions
