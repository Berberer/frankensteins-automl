import copy
import logging
from frankensteins_automl.search_space.search_space_component_instance import (
    SearchSpaceComponentInstance,
)


logger = logging.getLogger(__name__)


def construct_pipeline(start_component, rest_problem, parameter):
    component_mapping = copy.deepcopy(rest_problem.get_component_mapping())
    required_interfaces = copy.deepcopy(rest_problem.get_required_interfaces())
    for component_id, component in component_mapping.items():
        if component.get_name() == start_component:
            return assemble_component_instance(
                component_id, component_mapping, required_interfaces, parameter
            )
    logger.warn(f"No component found as start {start_component}")
    return None


def assemble_component_instance(
    component_id, component_mapping, required_interfaces, parameter
):
    logger.info(f"Assembling {component_id}")
    logger.info(f"Params for assembling: {parameter}")
    logger.info(f"Components for assembling: {component_mapping}")
    params = parameter[component_id]
    del parameter[component_id]
    component = component_mapping[component_id]
    del component_mapping[component_id]
    instance = SearchSpaceComponentInstance(component)
    # Create required interfaces
    if component.has_required_interfaces():
        # Get all interfaces for the current component
        interfaces = list(
            filter(
                lambda i: i["component_id"] == component_id,
                required_interfaces,
            )
        )
        # Create an instance for this component
        interface_instances = {}
        for interface in interfaces:
            required_interfaces.remove(interface)
            created_element = assemble_component_instance(
                interface["satisfied_with"],
                component_mapping,
                required_interfaces,
                parameter,
            )
            if created_element is not None:
                interface_instances[
                    interface["interface"]["id"]
                ] = created_element
            else:
                logger.warn(
                    f"No instance of {interface['name']} could be created"
                )
        if not instance.set_required_interfaces(interface_instances):
            logger.error(
                f"Created interfaces were invalid for {component.get_name()}"
            )
            return None
    # Set parameter and create element from configured instance
    if not instance.set_parameter_config(params):
        logger.error(
            f"Given parameter were invalid for f{component.get_name()}"
        )
        return None
    else:
        return instance.construct_pipeline_element()