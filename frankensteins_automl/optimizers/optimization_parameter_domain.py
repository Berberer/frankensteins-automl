import heapq
import logging
import uuid


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
        self.id_to_scores_mapping = {}
        self.id_to_configuration_mapping = {}
        self.configuration_string_to_id_mapping = {}

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
        config_str = str(config)
        if config_str not in self.configuration_string_to_id_mapping:
            config_id = str(uuid.uuid1())
            self.configuration_string_to_id_mapping[config_str] = config_id
            self.id_to_scores_mapping[config_id] = score
            self.id_to_configuration_mapping[config_id] = config
            heapq.heappush(self.results, (score, config_id))

    def get_top_results(self, top_n):
        results = heapq.nlargest(top_n, self.results)
        result_configurations = []
        for result in results:
            result_configurations.append(
                (result[0], self.id_to_configuration_mapping[result[1]])
            )
        return result_configurations

    def get_score_of_result(self, result):
        result_str = str(result)
        if result_str in self.configuration_string_to_id_mapping:
            configuration_id = self.configuration_string_to_id_mapping[
                result_str
            ]
            return self.id_to_scores_mapping[configuration_id]
        return None

    def has_results(self):
        return len(self.results) > 0
