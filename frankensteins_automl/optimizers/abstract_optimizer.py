import logging
import numpy
import random
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AbstractOptimizer(ABC):
    def __init__(
        self, parameter_domain, pipeline_evaluator, pipeline_evaluation_timeout
    ):
        super().__init__()
        self.parameter_domain = parameter_domain
        self.pipeline_evaluator = pipeline_evaluator
        self.pipeline_evaluation_timeout = pipeline_evaluation_timeout
        self.min_vector = self.parameter_domain.get_min_vector()
        self.max_vector = self.parameter_domain.get_max_vector()

    def _score_candidate(self, candidate):
        score = numpy.clip(candidate, self.min_vector, self.max_vector)
        score = self.parameter_domain.get_score_of_result(candidate)
        if score is None:
            configuration = self.parameter_domain.config_from_vector(candidate)
            score = self.pipeline_evaluator.evaluate_pipeline(
                configuration, timeout=self.pipeline_evaluation_timeout
            )
            self.parameter_domain.add_result(candidate, score)
        return score

    def _random_transform_candidate(self, candidate, number_of_changes):
        number_of_changes = min(len(candidate), number_of_changes)
        indices = random.sample(range((len(candidate))), number_of_changes)
        for index in indices:
            lower_bound = max(self.min_vector[index], candidate[index] - 0.5)
            upper_bound = min(self.max_vector[index], candidate[index] + 0.5)
            candidate[index] = random.uniform(lower_bound, upper_bound)
        return candidate

    @abstractmethod
    def perform_optimization(self, optimization_time_budget):
        pass
