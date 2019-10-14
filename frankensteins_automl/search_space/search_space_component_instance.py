import logging


logger = logging.getLogger(__name__)


class SearchSpaceComponentInstance(object):
    def __init__(self, component):
        self.component = component
        self.parameter = None

    def set_parameter_config(self, config):
        if self.component.validate_parameter_config(config):
            self.parameter = config
            return True
        logger.warning(
            f"Parameter {config} are not valid for {self.component.get_name()}"
        )
        return False

    def construct_pipeline_element(self):
        if self.component.has_parameter() and len(self.parameter) == 0:
            return None
        name_elements = self.component.get_name().split(".")
        module_path = name_elements[:-1]
        class_name = name_elements[-1]
        positional_parameter = []
        keyword_parameter = {}
        try:
            module = __import__(".".join(module_path), fromList=[class_name])
            component_constructor = getattr(module, class_name)
            logger.info(f"Imported constructor: {component_constructor}")
            return component_constructor(
                *positional_parameter, **keyword_parameter
            )
        except ImportError:
            logger.exception(
                f"Error while importing {self.component.get_name()}"
            )
        return None
