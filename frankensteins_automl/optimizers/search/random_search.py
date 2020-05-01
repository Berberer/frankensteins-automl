import logging
import numpy
from threading import Thread, Event
from frankensteins_automl.optimizers.abstract_optimizer import (
    AbstractOptimizer,
)

logger = logging.getLogger(__name__)


class RandomSearch(AbstractOptimizer):
    def __init__(
        self,
        parameter_domain,
        pipeline_evaluator,
        timeout_for_pipeline_evaluation,
        seed,
        numpy_random_state,
    ):
        super().__init__(
            parameter_domain,
            pipeline_evaluator,
            timeout_for_pipeline_evaluation,
            seed,
            numpy_random_state,
        )
        self.best_candidate = self.parameter_domain.get_default_config()
        self.best_score = self._score_candidate(self.best_candidate)
        self.stop_event = Event()

    def perform_optimization(self, optimization_time_budget):
        best_from_domain = self.parameter_domain.get_top_results(1)[0]
        self.best_score, self.best_candidate = best_from_domain
        logger.debug(f"Random Search starts with score: {self.best_score}")
        search_thread = Thread(target=self._search_loop)
        search_thread.start()
        search_thread.join(timeout=optimization_time_budget)
        self.stop_event.set()
        logger.debug(f"Random Search ends with score: {self.best_score}")
        return (
            self.parameter_domain.config_from_vector(self.best_candidate),
            self.best_score,
        )

    def _search_loop(self):
        while True:
            candidate = self._next_step()
            candidate_score = self._score_candidate(candidate)
            if candidate_score > self.best_score:
                self.best_candidate = candidate
                self.best_score = candidate_score
            if self.stop_event.is_set():
                logger.info("Stop event in random search")
                break

    def _next_step(self):
        candidate = numpy.copy(self.best_candidate)
        if len(candidate) == 0:
            return candidate
        self._random_transform_candidate(candidate, 1)
        return candidate
