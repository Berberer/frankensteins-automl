import importlib
import logging


logger = logging.getLogger(__name__)


class SearchSpaceComponentInstance(object):
    def __init__(self, component):
        self.component = component
        self.parameter = None
        self.required_interfaces = None

    def get_component(self):
        return self.component

    def set_parameter_config(self, config):
        if self.component.validate_parameter_config(config):
            self.parameter = config
            return True
        logger.warning(
            f"Parameter {config} are not valid for {self.component.get_name()}"
        )
        return False

    def set_required_interfaces(self, interface_elements):
        if self.component.has_required_interfaces():
            # Check if interface elements have as many members as needed
            if len(interface_elements) != len(
                self.component.get_required_interfaces()
            ):
                logger.info(
                    "Not valid because given interfaces have the wrong amount"
                )
                return False
            # Check given instances for each required interface
            for interface in self.component.get_required_interfaces():
                interface_id = interface["id"]
                interface_path = interface["name"]
                # Check if the given interface elements have this interface
                if interface_id not in interface_elements:
                    logger.info(f"No interface provided for id {interface_id}")
                    return False
                interface_element_path = (
                    interface_elements[interface_id].__class__.__module__
                    + "."
                    + interface_elements[interface_id].__class__.__name__
                )
                if interface_path != interface_element_path:
                    warning = str(
                        "For id "
                        + interface_id
                        + " component of type "
                        + interface_path
                        + " was expected but got an element of type "
                        + interface_element_path
                    )
                    logger.info(warning)
                    return False
            self.required_interfaces = interface_elements
            logger.info(
                f"{self.component.get_name()} interfaces: {interface_elements}"
            )
        return True

    def construct_pipeline_element(self):
        # Stop if parameters are needed and not provided
        if self.component.has_parameter() and (
            self.parameter is None or len(self.parameter) == 0
        ):
            return None
        # Stop if components are needed for required interfaces
        # but not provided
        if self.component.has_required_interfaces() and (
            self.required_interfaces is None
            or len(self.required_interfaces) == 0
        ):
            return None
        # Get path to element module and element constructor
        name_elements = self.component.get_name().split(".")
        module_path = name_elements[:-1]
        class_name = name_elements[-1]
        # Split parameters and required interfaces
        # into positional and keyword parameters
        args = self.component.create_construction_args_from_config(
            self.parameter, self.required_interfaces
        )
        positional_parameter, keyword_parameter = args
        # Import module and create element with parameters
        try:
            module = importlib.import_module(".".join(module_path))
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
