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

    async def perform_simulation(self, optimization_time_budget):
        current_node = self.start_node
        optimization_leaf = None
        while optimization_leaf is None:
            if current_node.is_leaf_node():
                optimization_leaf = current_node
            else:
                if current_node.get_successors() is None:
                    self.graph_generator.generate_successors()
                current_node = random.choice(current_node.get_successors())
        result, score = await optimization_leaf.perform_optimization(
            optimization_time_budget
        )
        return optimization_leaf, result, score


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
        while True:
            candidate_node = self._select_candidate_node()
            expanded_nodes = self._candidate_node_expansion(candidate_node)
            leaf_nodes = await self._simulation_of_expanded_nodes(
                expanded_nodes
            )
            self._back_propagation(leaf_nodes)

    def _select_candidate_node(self):
        current_node = self.root_node
        while (
            current_node.get_successors() is not None
            and len(current_node.get_successors()) > 0
        ):
            successors = current_node.get_successors()
            current_node = successors[0]
            best_score = current_node.get_node_value()
            for successor in successors:
                if successor.get_node_value() > best_score:
                    current_node = successor
                    best_score = successor.get_node_value()
        logger.info(f"Nex candidate for expansion is {current_node}")
        return current_node

    def _candidate_node_expansion(self, candidate_node):
        return self.graph_generator.generate_successors(candidate_node)

    async def _simulation_of_expanded_nodes(self, expanded_nodes):
        optimized_leafs = []
        random_searches = []
        for node in expanded_nodes:
            for i in range(self.config.simulation_runs_amount):
                random_searches.append(
                    RandomSearchSimulation(node, self.graph_generator)
                )
        logger.info(f"Start random search simulations")
        for simulation_result in asyncio.as_completed(
            [
                rs.perform_simulation(self.config.optimization_time_budget)
                for rs in random_searches
            ]
        ):
            leaf, result, score = await simulation_result
            leaf.recalculate_node_value(score)
            optimized_leafs.append(leaf)
            if score > self.best_score:
                logger.info(f"New best candidate is {result} with {score}")
                self.best_candidate = result
                self.best_score = score
        return optimized_leafs

    def _back_propagation(self, leaf_nodes):
        nodes = [(node, node.get_score_avg()) for node in leaf_nodes]
        while len(leaf_nodes) > 0:
            node, score = nodes.pop(0)
            node.recalculate_node_value(score)
            predecessor = node.get_predecessor()
            if predecessor is not None:
                node.append((predecessor, node.get_score_avg()))
