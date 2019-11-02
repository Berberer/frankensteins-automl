import heapq
import logging


logger = logging.getLogger(__name__)


class OptimizationParameterDomain(object):
    def __init__(self, component_mapping):
        self.component_mapping = component_mapping
        self.parameter_descriptions = {}
        for component_id, component in self.component_mapping.items():
            self.parameter_descriptions[
                component_id
            ] = component.get_parameter_description()
        self.results = []

    def get_parameter_descriptions(self):
        return self.parameter_descriptions

    def get_default_config(self):
        config = {}
        for component_id, component in self.component_mapping.items():
            config[component_id] = component.create_default_parameter_config()
        return config

    def draw_random_config(self):
        config = {}
        for component_id, component in self.component_mapping.items():
            config[component_id] = component.draw_random_parameter_config()
        return config

    def add_result(self, config, score):
        if (score, config) not in self.results:
            heapq.heappush(self.results, (score, config))

    def get_top_results(self, top_n):
        return heapq.nlargest(top_n, self.results)

    def has_results(self):
        return len(self.results) > 0
