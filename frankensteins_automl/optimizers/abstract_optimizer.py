import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AbstractOptimizer(ABC):
    def __init__(self, parameter_domain, pipeline_evaluator):
        super().__init__()
        self.parameter_domain = parameter_domain
        self.pipeline_evaluator = pipeline_evaluator

    def _score_candidate(self, candidate):
        score = self.parameter_domain.get_score_of_result(candidate)
        if score is None:
            score = self.pipeline_evaluator.evaluate_pipeline(candidate)
            self.parameter_domain.add_result(candidate, score)
        return score

    @abstractmethod
    def perform_optimization(self, optimization_time_budget):
        pass
