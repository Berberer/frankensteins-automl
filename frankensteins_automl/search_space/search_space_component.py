import copy
import logging


logger = logging.getLogger(__name__)


class SearchSpaceComponent(object):
    def __init__(self, description):
        self.name = description["name"]
        logger.info(f"Create search space component {self.name}")
        self.provided_interfaces = []
        if "providedInterface" in description:
            self.provided_interfaces = description["providedInterface"]
        self.required_interfaces = {}
        if "requiredInterface" in description:
            self.required_interfaces = description["requiredInterface"]
        self.params = {}
        for p in description["parameter"]:
            logger.info(f"Init param {p}")
            param_domain = copy.deepcopy(p)
            del param_domain["name"]
            self.params[p["name"]] = param_domain

    def get_name(self):
        return self.name

    def get_provided_interfaces(self):
        return self.provided_interfaces

    def get_required_interfaces(self):
        return self.required_interfaces

    def has_parameter(self):
        return len(self.params) > 0

    def has_required_interfaces(self):
        return len(self.required_interfaces) > 0

    def validate_parameter_config(self, config):
        logger.info(f"Validate param config of {self.name} with {config}")
        # Check if config has as many members as parameters needed
        if len(config) != len(self.params):
            logger.info("Not valid because config has wrong number of params")
            return False
        # Check given values for each param
        for param, domain in self.params.items():
            # Check if the config has this param
            if param in config:
                logger.info(f"Checking param {param}")
                param_type = domain["type"]
                # Type is a category
                if param_type == "cat":
                    # Check if the set value is one of the allowed categories
                    if config[param] not in domain["values"]:
                        logger.info(
                            f"Not valid because {config[param]} is unknown"
                        )
                        return False
                # Type is a number
                elif param_type == "double" or param_type == "int":
                    value = config[param]
                    # Check if value is smaller than allowed minimum
                    if value < domain["min"]:
                        logger.info("Not valid because value is too small")
                        return False
                    # Check if value is bigger than allowd maximum
                    if value > domain["max"]:
                        logger.info("Not valid because value is too big")
                        return False
                    # If type is double, check if value is float
                    if param_type == "double":
                        if not isinstance(value, float):
                            logger.info(
                                "Not valid because a double was expected"
                            )
                            return False
                    # If type is int, check if the value is int
                    elif param_type == "int":
                        if not isinstance(value, int):
                            logger.info(
                                "Not valid because an int was expected"
                            )
                            return False
                # One param type is unknown and cannot be validated
                else:
                    logger.warning(f"Unknown param type {param_type}")
            # Config is missing a paramter and is not valid
            else:
                logger.info(
                    f"Not valid because config does not include {param}"
                )
                return False
        # Validation of all config params was successfull
        return True
