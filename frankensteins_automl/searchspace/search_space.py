import logging

logger = logging.getLogger(__name__)


class SearchSpace(object):
    def __init__(self, components):
        self.components = components
        self.components_by_name = {}
        self.components_providing_interface = {}
        for component in components:
            component_name = component.get_name()
            provided_interfaces = component.get_provided_interfaces()
            self.components_by_name[component_name] = component
            for interface in provided_interfaces:
                if interface not in self.components_providing_interface:
                    self.components_providing_interface[interface] = []
                if (
                    component_name
                    not in self.components_providing_interface[interface]
                ):
                    self.components_providing_interface[interface].append(
                        component_name
                    )

    def get_components_providing_interface(self, interface_name):
        if interface_name in self.components_providing_interface:
            component_names = self.components_providing_interface[
                interface_name
            ]
            components = []
            for component in component_names:
                components.append(self.components_by_name[component])
            return components
        return None
