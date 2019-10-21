import asyncio
import logging
import random
from frankensteins_automl.mcts.mcts_search_graph import MctsGraphGenerator
from frankensteins_automl.search_space.search_space_reader import (
    create_search_space,
)

logger = logging.getLogger(__name__)


class RandomSearchSimulation:
    def __init__(self, start_node, graph_generator):
        self.start_node = start_node
        self.graph_generator = graph_generator

    def perform_simulation(self, optimization_time_budget):
        current_node = self.start_node
        optimization_leaf = None
        while optimization_leaf is None:
            if current_node.is_leaf_node():
                optimization_leaf = current_node
            else:
                if current_node.get_successors() is None:
                    self.graph_generator.generate_successors()
                current_node = random.choice(current_node.get_successors())
        score = optimization_leaf.perform_optimization_simulation(
            optimization_time_budget
        )
        return optimization_leaf, score


class MctsSearchConfig:
    def __init__(self):
        self.search_timeout = 600
        self.optimization_time_budget = 30
        self.search_space_files = [
            "res/search_space/ml-plan-ul.json",
            "res/search_space/scikit-learn-classifiers-tpot.json",
            "res/search_space/scikit-learn-preprocessors-tpot.json",
        ]
        self.start_component_name = "sklearn.pipeline.make_pipeline"
        self.warm_start_optimizers = True
        self.simulation_runs_amount = 3


class MctsSearch:
    def __init__(self, search_config, optimizers):
        self.config = search_config
        self.search_space = create_search_space(
            *self.config.search_space_files
        )
        self.graph_generator = MctsGraphGenerator(
            self.search_space, self.config.start_component_name, optimizers
        )
        self.root_node = self.graph_generator.get_root_node()
        self.search_front = [self.root_node]
        self.best_candidate = None
        self.best_score = 0.0

    async def run_search(self):
        try:
            await asyncio.wait_for(
                self._search_awaitable(), timeout=self.config.search_timeout
            )
        except asyncio.TimeoutError:
            logger.info(
                f"Search timeout. Best candidate had score {self.best_score}"
            )
        except Exception as e:
            logger.error(f"Error during search: {e}")
        return self.best_candidate, self.best_score

    async def _search_awaitable(self):
        candidate_node = self._select_candidate_node()
        expanded_nodes = self._candidate_node_expansion(candidate_node)
        leaf_nodes = self._simulation_of_expanded_nodes(expanded_nodes)
        self._back_propagation(leaf_nodes)

    def _select_candidate_node(self):
        pass

    def _candidate_node_expansion(self, candidate_node):
        pass

    def _simulation_of_expanded_nodes(self, expanded_nodes):
        pass

    def _back_propagation(self, leaf_nodes):
        pass
