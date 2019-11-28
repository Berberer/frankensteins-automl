import logging
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock, Event
from frankensteins_automl.event_listener import event_topics
from frankensteins_automl.mcts.mcts_search_graph import MctsGraphGenerator
from frankensteins_automl.machine_learning.pipeline import (
    pipeline_constructor,
    pipeline_evaluator,
)
from frankensteins_automl.search_space.search_space_reader import (
    create_search_space,
)

logger = logging.getLogger(__name__)
topic = event_topics.MCTS_TOPIC


graph_generation_lock = Lock()


class RandomSearchSimulation:
    def __init__(self, start_node, graph_generator):
        self.start_node = start_node
        self.graph_generator = graph_generator
        self.id = str(uuid.uuid1())

    def perform_simulation(self, optimization_time_budget):
        logger.debug(
            f"{self.id}-Start random search simulation at: {self.start_node}"
        )
        current_node = self.start_node
        optimization_leaf = None
        while optimization_leaf is None:
            if current_node.is_leaf_node():
                optimization_leaf = current_node
            else:
                successors = current_node.get_successors()
                if successors is None or successors == []:
                    graph_generation_lock.acquire()
                    successors = self.graph_generator.generate_successors(
                        current_node
                    )
                    graph_generation_lock.release()
                current_node = random.choice(successors)
        logger.debug(f"{self.id}-Optimize at leaf {optimization_leaf}")
        result, score = optimization_leaf.perform_optimization(
            optimization_time_budget
        )
        logger.debug(f"{self.id}-Optimization finished")
        return optimization_leaf, result, score


class MctsSearchConfig:
    def __init__(self, data_x, data_y):
        self.search_timeout = 600.0
        self.optimization_time_budget = 30.0
        self.timeout_for_pipeline_evaluation = 10.0
        self.pipeline_evaluator_class = pipeline_evaluator.PipelineEvaluator
        self.search_space_files = [
            "res/search_space/ml-plan-ul.json",
            "res/search_space/scikit-learn-classifiers-tpot.json",
            "res/search_space/scikit-learn-preprocessors-tpot.json",
        ]
        self.start_component_name = "sklearn.pipeline.make_pipeline"
        self.simulation_runs_amount = 3
        self.data_x = data_x
        self.data_y = data_y


class MctsSearch:
    def __init__(self, search_config, optimizers):
        self.config = search_config
        self.search_space = create_search_space(
            *self.config.search_space_files
        )
        self.graph_generator = MctsGraphGenerator(
            self.search_space,
            self.config.start_component_name,
            optimizers,
            self.config.pipeline_evaluator_class,
            self.config.timeout_for_pipeline_evaluation,
            self.config.data_x,
            self.config.data_y,
        )
        self.root_node = self.graph_generator.get_root_node()
        self.best_candidate = None
        self.best_score = 0.0
        self.stop_event = Event()

    def run_search(self):
        try:
            logger.info(
                f"Start MCTS with timeout: {self.config.search_timeout}"
            )
            logger.info(
                f"Optimization budget: {self.config.optimization_time_budget}"
            )
            logger.info(
                f"Simulations per leaf: {self.config.simulation_runs_amount}"
            )
            search_thread = Thread(target=self._search_loop, daemon=True)
            search_thread.start()
            search_thread.join(timeout=self.config.search_timeout)
            logger.info("Stopping MCTS ...")
            self.stop_event.set()
        except Exception as e:
            logger.error(f"Error during search: {e}")
        return self.best_candidate, self.best_score

    def _search_loop(self):
        while True:
            if self.stop_event.is_set():
                logger.debug("Stop event in MCTS loop")
                break
            logger.debug("Next MCTS search loop iteration")
            logger.debug("Select candidate node")
            candidate_node = self._select_candidate_node()
            logger.debug("Expand candidate node")
            expanded_nodes = self._candidate_node_expansion(candidate_node)
            logger.debug("Simulate expanded node")
            optimization_results = self._simulation_of_expanded_nodes(
                expanded_nodes
            )
            if self.stop_event.is_set():
                logger.debug("Stop event in MCTS loop")
                break
            logger.debug("Check results for new best candidate")
            leaf_nodes = self._check_optimization_results(optimization_results)
            logger.debug("Perform back propagation")
            self._back_propagation(leaf_nodes)
            if self.stop_event.is_set():
                logger.debug("Stop event in MCTS loop")
                break
        logger.debug("End of MCTS search loop reached")

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
        logger.debug(f"Next candidate for expansion is {current_node}")
        return current_node

    def _candidate_node_expansion(self, candidate_node):
        if candidate_node.is_leaf_node():
            return [candidate_node]
        graph_generation_lock.acquire()
        successors = self.graph_generator.generate_successors(candidate_node)
        graph_generation_lock.release()
        return successors

    def _simulation_of_expanded_nodes(self, expanded_nodes):
        def _run_random_search_simulation(start_node):
            random_search = RandomSearchSimulation(
                start_node, self.graph_generator
            )
            return random_search.perform_simulation(
                self.config.optimization_time_budget
            )

        start_nodes = expanded_nodes * self.config.simulation_runs_amount
        logger.debug(f"Starting simulations at: {start_nodes}")
        logger.debug(f"Start random search simulations")
        thread_pool = ThreadPoolExecutor()
        optimized_leafs = thread_pool.map(
            _run_random_search_simulation, start_nodes
        )
        thread_pool.shutdown(wait=False)
        logger.debug("Simulation threads mapped")
        return list(optimized_leafs)

    def _check_optimization_results(self, optimization_results):
        leaf_nodes = []
        for result in optimization_results:
            node, params, score = result
            leaf_nodes.append((node, score))
            if score > self.best_score:
                logger.info(f"Found better score: {score}")
                self.best_score = score
                self.best_candidate = pipeline_constructor.construct_pipeline(
                    self.config.start_component_name,
                    node.predecessor.get_rest_problem(),
                    params,
                )
                logger.info(f"New beste pipeline: {self.best_candidate}")
        return leaf_nodes

    def _back_propagation(self, leaf_nodes):
        while len(leaf_nodes) > 0:
            node, score = leaf_nodes.pop(0)
            logger.debug(f"Recalculate {node.get_node_id()} with {score}")
            node.recalculate_node_value(score)
            predecessor = node.get_predecessor()
            if predecessor is not None:
                leaf_nodes.append((predecessor, score))
