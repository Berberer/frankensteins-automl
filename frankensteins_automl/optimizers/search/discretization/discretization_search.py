import logging
from threading import Lock
from frankensteins_automl.optimizers.abstract_optimizer import (
    AbstractOptimizer,
)
from frankensteins_automl.search_space.graphs import GraphGenerator, GraphNode


logger = logging.getLogger(__name__)

graph_generation_lock = Lock()


class DiscretizationSearch(AbstractOptimizer):
    def __init__(
        self,
        parameter_domain,
        pipeline_evaluator,
        timeout_for_pipeline_evaluation,
    ):
        super().__init__(
            parameter_domain,
            pipeline_evaluator,
            timeout_for_pipeline_evaluation,
        )

    def perform_optimization(self, optimization_time_budget):
        pass


class DiscretizationGraphNode(GraphNode):
    def __init__(self, predecessor):
        super().__init__(predecessor)


class DiscretizationGraphGenerator(GraphGenerator):
    def __init__(self, parameter_domain):
        self.parameter_domain = parameter_domain

    def get_root_node(self):
        pass

    def get_node_successors(self, node):
        pass
