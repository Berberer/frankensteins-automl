import logging
import random
import uuid
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class MonteCarloSimulationRunner:
    def __init__(
        self,
        start_nodes,
        runs_amount,
        graph_generator,
        graph_generation_lock,
        leaf_score_method,
    ):
        self.start_nodes = start_nodes * runs_amount
        self.runs_amount = runs_amount
        self.graph_generator = graph_generator
        self.graph_generation_lock = graph_generation_lock
        self.leaf_score_method = leaf_score_method
        self.thread_pool = ThreadPoolExecutor()

    def run(self, timeout):
        def _run_random_searches(node):
            random_search_run = RandomSearchRun(
                node,
                self.graph_generator,
                self.graph_generation_lock,
                self.leaf_score_method,
                timeout,
            )
            return random_search_run.search()

        logger.debug(
            f"Starting Monte Carlo simulations at: {self.start_nodes}"
        )
        logger.debug(f"Start random searches")
        found_leafs = self.thread_pool.map(
            _run_random_searches, self.start_nodes
        )
        self.thread_pool.shutdown(wait=False)
        logger.debug("Monte Carlo simulation threads mapped")
        return list(found_leafs)


class RandomSearchRun:
    def __init__(
        self,
        start_node,
        graph_generator,
        graph_generation_lock,
        leaf_score_method,
        leaf_score_timeout,
    ):
        self.start_node = start_node
        self.graph_generator = graph_generator
        self.graph_generation_lock = graph_generation_lock
        self.leaf_score_method = leaf_score_method
        self.leaf_score_timeout = leaf_score_timeout
        self.id = str(uuid.uuid1())

    def search(self):
        logger.debug(
            f"{self.id}-Start random search simulation at: {self.start_node}"
        )
        current_node = self.start_node
        leaf = None
        while leaf is None:
            if current_node.is_leaf_node():
                leaf = current_node
            else:
                successors = current_node.get_successors()
                if successors is None or successors == []:
                    self.graph_generation_lock.acquire()
                    successors = self.graph_generator.generate_successors(
                        current_node
                    )
                    self.graph_generation_lock.release()
                current_node = random.choice(successors)
        logger.debug(f"Random Search {self.id}-Score at leaf {leaf}")
        result, score = self.leaf_score_method(leaf, self.leaf_score_timeout)
        logger.debug(f"Random Search {self.id}-Optimization finished")
        return leaf, result, score
