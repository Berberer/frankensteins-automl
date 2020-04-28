import logging
from threading import Lock
from time import time
from frankensteins_automl.mcts.monte_carlo_simulation_runner import (
    MonteCarloSimulationRunner,
)
from frankensteins_automl.optimizers.abstract_optimizer import (
    AbstractOptimizer,
)
from frankensteins_automl.optimizers.search.discretization import (
    discretization_helper,
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
        numpy_random_state,
    ):
        super().__init__(
            parameter_domain,
            pipeline_evaluator,
            timeout_for_pipeline_evaluation,
            numpy_random_state,
        )
        self.best_candidate = self.parameter_domain.get_default_config()
        self.best_score = self._score_candidate(self.best_candidate)
        self.graph_generator = DiscretizationGraphGenerator(
            self.parameter_domain
        )

    def perform_optimization(self, optimization_time_budget):
        def score_atomic_discretization(node, _):
            config_vector = self.parameter_domain.config_to_vector(
                node.get_discretization().get_config()
            )
            return config_vector, self._score_candidate(config_vector)

        optimization_start = time()
        while (
            (time() - optimization_start) + self.pipeline_evaluation_timeout
        ) <= optimization_time_budget:
            # Stop optimization if the root node is covered
            # since then the whole graph was evaluated
            current_node = self.graph_generator.get_root_node()
            if current_node.is_covered():
                break

            # Find unexpanded, not covered node
            # where the highest rated sub-graph is rooted
            while not current_node.get_successors() == []:
                successors = current_node.get_successors()
                successors = list(
                    filter(lambda n: not n.is_covered(), successors)
                )
                best_successor = successors[0]
                best_successor_score = successors[0].get_best_successor_score()
                for successor in successors:
                    successor_score = successor.get_best_successor_score()
                    if successor_score > best_successor_score:
                        best_successor = successor
                        best_successor_score = successor_score
                current_node = best_successor

            # Expand this node
            graph_generation_lock.acquire()
            expanded_nodes = self.graph_generator.generate_successors(
                current_node
            )
            current_node.set_successors(expanded_nodes)
            graph_generation_lock.release()

            # Score expanded nodes with Monte Carlo simulations
            runner = MonteCarloSimulationRunner(
                expanded_nodes,
                self.config.simulation_runs_amount,
                self.graph_generator,
                graph_generation_lock,
                score_atomic_discretization,
            )

            # Check if the simulations found a new best candidate
            # and backpropagate the results from the leafs
            results = runner.run(self.config.optimization_time_budget)
            for result in results:
                leaf, candidate, score = result
                leaf.best_successor_score = score
                leaf.backpropagate()
                if score > self.best_score:
                    self.best_score = score
                    self.best_candidate = candidate
        return (
            self.parameter_domain.config_from_vector(self.best_candidate),
            self.best_score,
        )


class DiscretizationGraphNode(GraphNode):
    def __init__(self, predecessor, discretization):
        super().__init__(predecessor)
        self.discretization = discretization
        self.covered = self.discretization.is_atomic()
        self.best_successor_score = 0.0

    def backpropagate(self):
        if not self.covered:
            all_successores_covered = True
            for successor in self.successors:
                if not successor.is_covered():
                    all_successores_covered = False
                successor_score = successor.get_best_successor_score()
                if successor_score > self.best_successor_score:
                    self.best_successor_score = successor_score
            self.covered = all_successores_covered
            self.predecessor.backpropagate()

    def get_discretization(self):
        return self.discretization

    def is_covered(self):
        return self.covered

    def get_best_successor_score(self):
        return self.best_successor_score

    def is_leaf_node(self):
        return self.discretization.is_atomic()


class DiscretizationGraphGenerator(GraphGenerator):
    def __init__(self, parameter_domain):
        self.parameter_domain = parameter_domain
        self.root_node = DiscretizationGraphNode(
            None, discretization_helper.Discretization(parameter_domain)
        )

    def get_root_node(self):
        return self.root_node

    def get_node_successors(self, node):
        refinements = discretization_helper.refine_discretization(
            node.get_discretization()
        )
        successors = []
        for refinement in refinements:
            successors.append(DiscretizationGraphNode(node, refinement))
        return successors
