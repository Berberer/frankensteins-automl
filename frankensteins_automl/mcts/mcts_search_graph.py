import logging
import math
import uuid
from frankensteins_automl.search_space.search_space_graph import (
    SearchSpaceGraphNode,
    SearchSpaceGraphGenerator,
)

logger = logging.getLogger(__name__)


class MctsGraphNode(SearchSpaceGraphNode):
    def __init__(self, predecessor, rest_problem, optimizer):
        super().__init__(predecessor, rest_problem)
        self.optimizer = optimizer
        self.node_value = 0
        self.simulation_visits = 0
        self.score_avg = 0
        self.leaf_id = None

    def is_search_space_leaf_node(self):
        return super.is_leaf_node()

    def is_leaf_node(self):
        if super().is_leaf_node():
            return self.optimizer is not None
        else:
            False

    def get_node_value(self):
        return self.node_value

    def get_score_avg(self):
        return self.score_avg

    def get_simulation_visits(self):
        return self.simulation_visits

    def get_leaf_id(self):
        return self.leaf_id

    def set_leaf_id(self, leaf_id):
        self.leaf_id = leaf_id

    def recalculate_node_value(self, new_result):
        self.simulation_visits = self.simulation_visits + 1
        self.score_avg = self.score_avg + (
            (new_result - self.score_avg) / self.simulation_visits
        )
        exploration_factor = 1.41421 * math.sqrt(
            math.log(float(self.predecessor.get_simulation_visits()))
            / float(self.simulation_visits)
        )
        self.node_value = self.score_avg + exploration_factor
        pass

    async def perform_optimization(self, time_budget):
        if self.optimizer is not None:
            return self.optimizer.perform_optimization(
                time_budget, self.predecessor.get_leaf_id()
            )
        else:
            logger.warn("Node has not optimizer!")


class MctsGraphGenerator(SearchSpaceGraphGenerator):
    def __init__(
        self,
        search_space,
        initial_component_name,
        optimization_domain_store,
        optimizers,
    ):
        super().__init__(search_space, initial_component_name)
        self.optimizers = optimizers
        self.optimization_domain_store = optimization_domain_store

    def generate_successors(self, node):
        if node.is_leaf_node:
            return []
        else:
            successors = []
            if node.is_search_space_leaf_node():
                leaf_id = str(uuid.uuid1())
                node.set_leaf_id(leaf_id)
                self.optimization_domain_store.init_new_leaf(leaf_id)
                for optimizer in self.optimizers:
                    successors.append(
                        MctsGraphNode(node, node.get_rest_problem(), optimizer)
                    )
                logger.info(f"{len(successors)} successors for optimizers")
            else:
                successors = map(
                    lambda n: MctsGraphNode(
                        node, n.get_rest_problem(), None, None
                    ),
                    super().generate_successors(node),
                )
            node.set_successors(successors)
            return successors
