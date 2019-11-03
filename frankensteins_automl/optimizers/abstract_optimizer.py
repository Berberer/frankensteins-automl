import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AbstractOptimizer(ABC):
    def __init__(self, parameter_domain, pipeline_evaluator):
        super().__init__()
        self.parameter_domain = parameter_domain
        self.pipeline_evaluator = pipeline_evaluator

    @abstractmethod
    def perform_optimization(self, optimization_time_budget):
        pass
