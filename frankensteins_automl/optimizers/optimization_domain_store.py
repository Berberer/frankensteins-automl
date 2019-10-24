import logging


logger = logging.getLogger(__name__)


class ParameterDomain(object):
    def __init__(self, component_mapping, parameter_descriptions):
        self.component_mapping = component_mapping
        self.parameter_descriptions = parameter_descriptions

    def get_default_config(self):
        config = []
        for component_id, component in self.component_mapping.items():
            config[component_id] = component.create_default_parameter_config()
        return config

    def draw_random_config(self):
        config = []
        for component_id, component in self.component_mapping.items():
            config[component_id] = component.draw_random_parameter_config()
        return config

    def add_results(self, config, score):
        pass


class OptimizationResultStore(object):
    def __init__(self):
        self.domain_store = {}
        self.components_for_leaf = {}

    def has_leaf_id(self, leaf_id):
        return leaf_id in self.domain_store

    def init_new_leaf(self, leaf_node):
        leaf_id = leaf_node.get_leaf_id()
        if leaf_id is None:
            logger.warning(
                "Tried to init a result store with a non-leaf node!"
            )
        elif leaf_id in self.domain_store:
            logger.info(f"Leaf id {leaf_id} is already in result store")
        else:
            component_mapping = (
                leaf_node.get_rest_problem().get_component_mapping()
            )
            self.components_for_leaf[leaf_id] = component_mapping
            parameter_descriptions = {}
            for component_id, component in component_mapping.items():
                parameter_descriptions[
                    component_id
                ] = component.get_parameter_description()
            self.domain_store[leaf_id] = ParameterDomain(
                component_mapping, parameter_descriptions
            )

    def get_parameter_domain(self, leaf_id):
        if leaf_id in self.domain_store:
            return self.domain_store[leaf_id]
        else:
            logger.warning(f"Leaf id {leaf_id} does not exist in store")
            return None

    def add_results(self, leaf_id, config, score):
        if leaf_id in self.domain_store:
            self.domain_store[leaf_id].add_results(config, score)
        else:
            logger.warning(f"Leaf id {leaf_id} does not exist in store")
