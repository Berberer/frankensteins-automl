import logging
import numpy
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AbstractOptimizer(ABC):
    def __init__(self, parameter_domain, pipeline_evaluator):
        super().__init__()
        self.parameter_domain = parameter_domain
        self.pipeline_evaluator = pipeline_evaluator
        self.min_vector = self.parameter_domain.get_min_vector()
        self.max_vector = self.parameter_domain.get_max_vector()

    def _score_candidate(self, candidate):
        score = numpy.clip(candidate, self.min_vector, self.max_vector)
        score = self.parameter_domain.get_score_of_result(candidate)
        if score is None:
            configuration = self.parameter_domain.config_from_vector(candidate)
            score = self.pipeline_evaluator.evaluate_pipeline(configuration)
            self.parameter_domain.add_result(candidate, score)
        return score

    @abstractmethod
    def perform_optimization(self, optimization_time_budget):
        pass
